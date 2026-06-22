"""The structural mock provider.

Honesty matters here. A mock that returns a fixed fixture regardless of the
prompt would make the offline demo a lie: the optimizer "rewrites prompts" but
nothing changes. So this mock is a pure function of the *structural properties*
of the prompt — NOT of specific magic strings:

  - does the draft prompt ask the writer to respect a reading level?
  - does it ask for citations to fact ids?
  - does it ask to address every learning objective?

When a property is present, the mock produces output that genuinely satisfies
more eval criteria (matches the target reading level, embeds [fact_*] citations,
tags [obj_*] coverage). The causal chain — better-structured prompt -> better-
structured output -> higher deterministic scores — mirrors a plausible real
relationship and *generalizes across items* (it can't memorize an answer key,
because it responds to structure, not vocabulary). It is an honest simulation of
the pipeline, not a claim about real-model pedagogy. The Claude provider is where
real quality is judged.
"""

from __future__ import annotations

import hashlib
import re

from lesson_loom.providers.base import LLMProvider, LLMResponse

# Structural feature detectors — families of phrasing, not single magic strings.
_READING_LEVEL = re.compile(
    r"reading level|grade[- ]?level|target level|age[- ]?appropriate|"
    r"plain language|simple sentences|short sentences",
    re.I,
)
_CITATION = re.compile(r"\bcite\b|citation|\bsource(s)?\b|\breference|\[fact", re.I)
_OBJECTIVES = re.compile(
    r"every objective|each objective|all objectives|address the objectives|"
    r"learning objectives|\[obj", re.I
)

# Short, low-grade sentence scaffolding (Flesch-Kincaid ~ grade 4-6).
# Every template carries a {C} slot so each cited fact gets a home.
_SIMPLE = [
    "This important idea is fun and easy to learn. {C}",
    "Let us look at how it really works. {C}",
    "Plants and animals depend on it every day. {C}",
    "We can see it happen all around us. {C}",
    "Here is another useful fact to remember. {C}",
    "Now you understand a little bit more. {C}",
]
# Long, high-grade sentence scaffolding (Flesch-Kincaid ~ grade 13-17).
_COMPLEX = [
    "The multifaceted phenomenon of {T} necessitates comprehensive analytical "
    "consideration across numerous interrelated conceptual dimensions. {C}",
    "Consequently, an examination of the underlying mechanisms reveals "
    "considerable theoretical sophistication and intricate interdependencies. {C}",
    "Furthermore, the aforementioned principles substantiate the proposition that "
    "rigorous interrogation yields meaningful epistemological consequences. {C}",
    "Notwithstanding such complexity, systematic deconstruction facilitates a more "
    "nuanced appreciation of the constituent elements. {C}",
]
# Mid-grade scaffolding (Flesch-Kincaid ~ grade 9-12) for secondary students.
_MEDIUM = [
    "The new machines slowly changed the way that people produced goods and the "
    "way that they lived and worked across the growing towns. {C}",
    "Over several decades, many families moved into the expanding cities to find "
    "factory work, and the towns grew larger and more crowded. {C}",
    "Workers often spent very long hours inside the new factories, and their daily "
    "lives changed in ways that were difficult and frequently unfair. {C}",
    "These developments did not happen suddenly, but gradually accumulated over "
    "many years across the different regions of the country. {C}",
]


def _select_tier(low: float, high: float):
    mid = (low + high) / 2.0
    if mid <= 6.5:
        return _SIMPLE
    if mid <= 12.0:
        return _MEDIUM
    return _COMPLEX


def _parse_block(user: str, label: str) -> list[tuple[str, str]]:
    """Parse a 'LABEL:' section of '- id: text' lines from the user message."""
    out: list[tuple[str, str]] = []
    in_section = False
    for line in user.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith(label + ":"):
            in_section = True
            continue
        if in_section:
            if stripped.startswith("-") and ":" in stripped:
                body = stripped[1:].strip()
                key, _, val = body.partition(":")
                out.append((key.strip(), val.strip()))
            elif stripped and not stripped.startswith("-"):
                break
    return out


def _parse_reading_target(user: str) -> tuple[float, float]:
    m = re.search(r"READING_TARGET:\s*([\d.]+)\s*-\s*([\d.]+)", user)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 4.0, 6.5


def _scalar(field: str, user: str, default: str = "") -> str:
    m = re.search(rf"{field}:\s*(.+)", user)
    return m.group(1).strip() if m else default


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, seed: int = 42):
        self.seed = seed

    # -- free-form completion --------------------------------------------------

    def complete(self, system: str, user: str, **kwargs) -> LLMResponse:
        node = kwargs.get("node", "")
        if node == "draft":
            text = self._draft(system, user)
        elif node == "plan":
            text = self._plan(user)
        elif node == "critique":
            text = "ACCEPT"  # deterministic single-pass in mock mode
        elif node == "revise":
            text = self._draft(system, user)  # revise re-drafts under the same prompt
        else:
            text = f"[mock:{node}]"
        return LLMResponse(text=text, model="mock", input_tokens=len(user), output_tokens=len(text))

    def _plan(self, user: str) -> str:
        topic = _scalar("TOPIC", user, "the topic")
        return f"1. Intro to {topic}\n2. Key facts\n3. Examples\n4. Summary"

    def _draft(self, system: str, user: str) -> str:
        wants_reading = bool(_READING_LEVEL.search(system))
        wants_citations = bool(_CITATION.search(system))
        wants_objectives = bool(_OBJECTIVES.search(system))

        topic = _scalar("TOPIC", user, "the topic")
        low, high = _parse_reading_target(user)
        facts = _parse_block(user, "FACTS")
        objectives = _parse_block(user, "OBJECTIVES")

        # Reading level: only matches the target band when the prompt asks for it.
        # Otherwise the writer "writes naturally" (complex prose) regardless of grade.
        if wants_reading:
            templates = _select_tier(low, high)
        else:
            templates = _COMPLEX

        # Citations: embed [fact_*] tags only when the prompt requests grounding.
        cited = [fid for fid, _ in facts] if wants_citations else []

        sentences: list[str] = []
        for i, fid in enumerate(cited or [None] * max(len(facts), 3)):
            tmpl = templates[i % len(templates)]
            cite = f"[{fid}]" if fid else ""
            sentences.append(tmpl.format(T=topic, C=cite).replace("  ", " ").strip())

        body = " ".join(sentences)

        # Objective coverage: tag all objectives when asked, else just the first.
        covered = [oid for oid, _ in objectives] if wants_objectives else (
            [objectives[0][0]] if objectives else []
        )
        if covered:
            tags = " ".join(f"[{oid}]" for oid in covered)
            body += f"\n\nObjectives addressed: {tags}"

        return body

    # -- structured judge ------------------------------------------------------

    def judge(self, system: str, user: str, **kwargs) -> dict:
        """Deterministic but rubric- and feature-aware. A sensible judge rates
        cited, objective-covering, well-sized content higher. Engagement is keyed
        on interactive signals (a question or a worked example) that the templated
        mock never produces — so engagement stays a genuine, persistent weakness
        the optimizer must reckon with (try to fix, fail, then instrument)."""
        body = _scalar_multiline("BODY", user)
        rubric = system.lower()
        has_citations = "[fact_" in body
        covers_multi = body.count("[obj_") >= 2
        sized_ok = 40 <= len(body) <= 2000
        jitter = (int(hashlib.sha256(f"{self.seed}:{body}".encode()).hexdigest(), 16) % 2) * 0.1

        if "engaging" in rubric:
            engaging = ("?" in body) or ("for example" in body.lower())
            base = 2.0 + (2.0 if engaging else 0.0)
        elif "clear" in rubric:
            base = 2.0 + 1.3 * has_citations + 1.3 * sized_ok
        else:  # pedagogical soundness
            base = 2.0 + 1.2 * has_citations + 0.9 * covers_multi + 0.4 * sized_ok

        score = max(1, min(5, round(base + jitter)))
        return {"score": int(score), "rationale": "mock judge (rubric- and feature-aware)"}


def _scalar_multiline(field: str, user: str) -> str:
    """Grab everything after 'FIELD:' to end-of-message (judge passes BODY last)."""
    idx = user.find(field + ":")
    return user[idx + len(field) + 1 :].strip() if idx >= 0 else user

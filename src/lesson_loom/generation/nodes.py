"""Generation graph nodes.

Each node is a pure function of its input state slice and returns only the keys
it changes — the property that makes the whole graph deterministic. The provider,
context pack, and system config travel in the state (the graph is compiled
without a checkpointer, so nothing is serialized between runs)."""

from __future__ import annotations

import re

from lesson_loom.core.schemas import CitationLink, ContentArtifact
from lesson_loom.generation import prompts

_FACT_TAG = re.compile(r"\[(fact_\w+)\]")
_OBJ_TAG = re.compile(r"\[(obj_\w+)\]")


def plan_node(state: dict) -> dict:
    cfg = state["system_config"]
    user = prompts.build_plan_user(state["topic"], state["content_type"], state["grade_level"])
    resp = state["provider"].complete(cfg.generation_prompts["plan"], user, node="plan")
    return {"outline": resp.text}


def retrieve_node(state: dict) -> dict:
    """Code-only grounding: pull the topic's facts + objectives from the pack."""
    pack = state["context_pack"]
    return {"retrieved_facts": pack.facts, "retrieved_objectives": pack.learning_objectives}


def _draft(state: dict, system: str) -> str:
    user = prompts.build_draft_user(
        state["topic"],
        state["content_type"],
        state["grade_level"],
        state["reading_low"],
        state["reading_high"],
        state["retrieved_facts"],
        state["retrieved_objectives"],
    )
    return state["provider"].complete(system, user, node="draft").text


def draft_node(state: dict) -> dict:
    cfg = state["system_config"]
    return {"draft_text": _draft(state, cfg.generation_prompts["draft"]), "revision_count": 0}


def self_critique_node(state: dict) -> dict:
    cfg = state["system_config"]
    resp = state["provider"].complete(
        cfg.generation_prompts["critique"], f"BODY: {state['draft_text']}", node="critique"
    )
    return {"critique": resp.text}


def revise_node(state: dict) -> dict:
    cfg = state["system_config"]
    return {
        "draft_text": _draft(state, cfg.generation_prompts["revise"]),
        "revision_count": state["revision_count"] + 1,
    }


def format_node(state: dict) -> dict:
    body = state["draft_text"]
    cfg = state["system_config"]

    citations: list[CitationLink] = []
    seen: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        for fid in _FACT_TAG.findall(sentence):
            if fid not in seen:
                seen.add(fid)
                citations.append(CitationLink(fact_id=fid, excerpt=sentence.strip()))

    objectives_covered = list(dict.fromkeys(_OBJ_TAG.findall(body)))

    artifact = ContentArtifact.build(
        system_config_id=cfg.id,
        topic=state["topic"],
        content_type=state["content_type"],
        grade_level=state["grade_level"],
        body=body,
        citations=citations,
        objectives_covered=objectives_covered,
    )
    return {"artifact": artifact}


def route_after_critique(state: dict) -> str:
    cfg = state["system_config"]
    if "REVISE" in (state.get("critique") or "").upper() and state["revision_count"] < cfg.max_revisions:
        return "revise"
    return "accept"

"""Real Claude provider (optional — `pip install lesson-loom[claude]` + ANTHROPIC_API_KEY).

Uses the stable Anthropic Messages API. The judge uses a forced tool call so the
rubric score comes back as structured JSON rather than free text. Generation
defaults to claude-sonnet-4-6; the judge defaults to claude-opus-4-8.
"""

from __future__ import annotations

import json

from lesson_loom.providers.base import LLMProvider, LLMResponse

_JUDGE_TOOL = {
    "name": "record_rubric_score",
    "description": "Record a 1-5 rubric score with a one-sentence rationale.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "rationale": {"type": "string"},
        },
        "required": ["score", "rationale"],
    },
}


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, generation_model: str, judge_model: str):
        import anthropic  # imported lazily so the core has no hard dependency

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.generation_model = generation_model
        self.judge_model = judge_model

    def complete(self, system: str, user: str, **kwargs) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.generation_model,
            max_tokens=kwargs.get("max_tokens", 2048),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResponse(
            text=text,
            model=self.generation_model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    def judge(self, system: str, user: str, **kwargs) -> dict:
        resp = self.client.messages.create(
            model=self.judge_model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[_JUDGE_TOOL],
            tool_choice={"type": "tool", "name": "record_rubric_score"},
        )
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use":
                return {"score": int(block.input["score"]), "rationale": block.input["rationale"]}
        # Defensive fallback: try to parse a JSON object from any text.
        text = "".join(getattr(b, "text", "") for b in resp.content)
        try:
            data = json.loads(text[text.find("{") : text.rfind("}") + 1])
            return {"score": int(data["score"]), "rationale": data.get("rationale", "")}
        except Exception:
            return {"score": 3, "rationale": "unparseable judge response"}

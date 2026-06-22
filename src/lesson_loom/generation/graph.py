"""The LangGraph generation graph: plan -> retrieve -> draft -> self_critique
-> (revise loop | format) -> END.

Compiled without a checkpointer: each generation is a fresh, stateless run, which
is exactly what reproducibility wants (and sidesteps thread-id/checkpoint reuse
bugs). Determinism comes from pure nodes + the deterministic provider."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from lesson_loom.core.schemas import ContentArtifact, ContextPack, SystemConfig
from lesson_loom.generation import nodes
from lesson_loom.providers.base import LLMProvider


class GenerationState(TypedDict, total=False):
    # inputs
    topic: str
    grade_level: str
    content_type: str
    reading_low: float
    reading_high: float
    context_pack: ContextPack
    system_config: SystemConfig
    provider: LLMProvider
    # working
    outline: str
    retrieved_facts: list
    retrieved_objectives: list
    draft_text: str
    critique: str
    revision_count: int
    # output
    artifact: ContentArtifact


def build_generation_graph():
    g = StateGraph(GenerationState)
    g.add_node("plan", nodes.plan_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("draft", nodes.draft_node)
    g.add_node("self_critique", nodes.self_critique_node)
    g.add_node("revise", nodes.revise_node)
    g.add_node("format", nodes.format_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "self_critique")
    g.add_edge("revise", "self_critique")
    g.add_conditional_edges(
        "self_critique", nodes.route_after_critique, {"revise": "revise", "accept": "format"}
    )
    g.add_edge("format", END)
    return g.compile()


# Built once; the graph is stateless so it is safe to reuse across calls.
_GRAPH = build_generation_graph()


def generate(
    *,
    provider: LLMProvider,
    config: SystemConfig,
    context_pack: ContextPack,
    topic: str,
    grade_level: str,
    content_type: str,
    reading_low: float,
    reading_high: float,
) -> ContentArtifact:
    initial: dict[str, Any] = {
        "topic": topic,
        "grade_level": grade_level,
        "content_type": content_type,
        "reading_low": reading_low,
        "reading_high": reading_high,
        "context_pack": context_pack,
        "system_config": config,
        "provider": provider,
        "revision_count": 0,
    }
    final = _GRAPH.invoke(initial)
    return final["artifact"]

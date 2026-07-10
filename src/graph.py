from time import perf_counter
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from adjudicator import Adjudicator, AdjudicatorFailureException
from critics.base import BaseCritic, CriticFailureException
from critics.completeness import CompletenessCritic
from critics.factual_accuracy import FactualAccuracyCritic
from critics.logical_consistency import LogicalConsistencyCritic
from disagreement import detect_disagreements
from schemas import ArbitrationResult, Critique, Verdict


def merge_critiques(existing: list, update: list) -> list:
    if update and isinstance(update[0], dict) and update[0].get("__replace__"):
        return update[0]["items"]
    return existing + update


def merge_errors(existing: list, update: list) -> list:
    return existing + update


class ArbitrationState(TypedDict):
    input_text: str
    original_prompt: str
    critiques: Annotated[list, merge_critiques]
    verdict: dict
    errors: Annotated[list, merge_errors]
    critics_available: int
    graph_started_at: float
    adjudicator_tokens_used: int
    adjudicator_model_used: str
    adjudicator_latency_ms: float
    arbitration_result: dict


def dispatch(state: ArbitrationState) -> dict[str, Any]:
    print("Dispatching critics in parallel.")
    return {"graph_started_at": state.get("graph_started_at") or perf_counter()}


def run_factual_critic(state: ArbitrationState) -> dict[str, list]:
    try:
        critique = FactualAccuracyCritic().critique(state["input_text"])
        print("Completed factual_accuracy critique:")
        print(critique.model_dump_json(indent=2))
        return {"critiques": [critique.model_dump()]}
    except CriticFailureException as exc:
        return {
            "errors": [
                {"critic_type": exc.critic_type, "error": str(exc)}
            ]
        }


def run_logical_critic(state: ArbitrationState) -> dict[str, list]:
    try:
        critique = LogicalConsistencyCritic().critique(state["input_text"])
        print("Completed logical_consistency critique:")
        print(critique.model_dump_json(indent=2))
        return {"critiques": [critique.model_dump()]}
    except CriticFailureException as exc:
        return {
            "errors": [
                {"critic_type": exc.critic_type, "error": str(exc)}
            ]
        }


def run_completeness_critic(state: ArbitrationState) -> dict[str, list]:
    try:
        critique = CompletenessCritic().critique(
            text=state["input_text"],
            original_prompt=state.get("original_prompt", ""),
        )
        print("Completed completeness critique:")
        print(critique.model_dump_json(indent=2))
        return {"critiques": [critique.model_dump()]}
    except CriticFailureException as exc:
        return {
            "errors": [
                {"critic_type": exc.critic_type, "error": str(exc)}
            ]
        }


def collect(state: ArbitrationState) -> dict[str, Any]:
    sorted_critiques = sorted(
        state["critiques"],
        key=lambda critique: critique["critic_type"],
    )
    critics_available = len(sorted_critiques)
    critic_failures = len(state["errors"])

    if critic_failures == 3:
        print("All critics failed. Arbitration unavailable.")
        return {
            "critiques": [{"__replace__": True, "items": []}],
            "critics_available": 0,
            "verdict": {
                "quality_score": 0,
                "confidence": 0,
                "confirmed_issues": [],
                "dismissed_flags": [],
                "critics_agreed": False,
                "summary": "All critics failed — arbitration unavailable",
                "critics_failed": True,
                "error_count": critic_failures,
            },
        }

    if critic_failures:
        print(f"Collected critiques with {critic_failures} critic failure(s).")
    else:
        print("Collected all critiques successfully.")

    average_score = sum(
        critique["score"] for critique in sorted_critiques
    ) / critics_available
    quality_score = round(average_score * 2)

    if critic_failures == 0:
        confidence = 4
        summary = "Arbitration completed with all critics available."
    elif critic_failures == 1:
        confidence = 3
        summary = "Arbitration completed with one critic unavailable."
    else:
        confidence = 1
        summary = "Arbitration completed with very low confidence because two critics failed."

    return {
        "critiques": [{"__replace__": True, "items": sorted_critiques}],
        "critics_available": critics_available,
        "verdict": {
            "quality_score": quality_score,
            "confidence": confidence,
            "confirmed_issues": [
                issue
                for critique in sorted_critiques
                for issue in critique.get("issues", [])
            ],
            "dismissed_flags": [],
            "critics_agreed": critic_failures == 0,
            "summary": summary,
            "critics_failed": critic_failures > 0,
            "error_count": critic_failures,
        },
    }


def adjudicate(state: ArbitrationState) -> dict[str, Any]:
    if state["critics_available"] == 0:
        print("Skipping adjudicator because no critics succeeded.")
        return {}

    critiques = [
        critique if isinstance(critique, Critique) else Critique(**critique)
        for critique in state["critiques"]
    ]
    disagreements = detect_disagreements(critiques)

    try:
        adjudicator = Adjudicator()
        verdict = adjudicator.adjudicate(
            input_text=state["input_text"],
            critiques=critiques,
            disagreements=disagreements,
        )
        verdict_data = verdict.model_dump()
        verdict_data["critics_failed"] = len(state["errors"]) > 0
        verdict_data["error_count"] = len(state["errors"])
        verdict_data["disagreement_count"] = len(disagreements)
        return {
            "verdict": verdict_data,
            "adjudicator_tokens_used": adjudicator.last_tokens_used,
            "adjudicator_model_used": adjudicator.model_name,
            "adjudicator_latency_ms": adjudicator.last_latency_ms,
        }
    except AdjudicatorFailureException as exc:
        return {
            "errors": [
                {"critic_type": "adjudicator", "error": str(exc)}
            ]
        }


def calculate_totals(state: ArbitrationState) -> dict[str, Any]:
    critiques = [
        critique if isinstance(critique, Critique) else Critique(**critique)
        for critique in state["critiques"]
    ]
    verdict = Verdict(**state["verdict"])

    critique_tokens = sum(critique.tokens_used for critique in critiques)
    adjudicator_tokens = state.get("adjudicator_tokens_used", 0)
    total_tokens_used = critique_tokens + adjudicator_tokens

    critique_cost = sum(
        BaseCritic._calculate_cost(critique.tokens_used, critique.model_used)
        for critique in critiques
    )
    adjudicator_cost = BaseCritic._calculate_cost(
        adjudicator_tokens,
        state.get("adjudicator_model_used", "deepseek-reasoner"),
    )
    total_cost_usd = critique_cost + adjudicator_cost
    total_latency_ms = (perf_counter() - state["graph_started_at"]) * 1000

    result = ArbitrationResult(
        input_text=state["input_text"],
        critiques=critiques,
        verdict=verdict,
        total_latency_ms=total_latency_ms,
        total_tokens_used=total_tokens_used,
        total_cost_usd=total_cost_usd,
    )

    print(
        f"Tokens: {total_tokens_used} | "
        f"Cost: ${total_cost_usd:.6f} | "
        f"Latency: {total_latency_ms:.0f}ms"
    )

    return {
        "arbitration_result": result.model_dump(),
    }


# Sequential version kept for reference:
#
# def build_sequential_arbitration_graph():
#     graph_builder = StateGraph(ArbitrationState)
#
#     graph_builder.add_node("run_factual_critic", run_factual_critic)
#     graph_builder.add_node("run_logical_critic", run_logical_critic)
#     graph_builder.add_node("run_completeness_critic", run_completeness_critic)
#
#     graph_builder.add_edge(START, "run_factual_critic")
#     graph_builder.add_edge("run_factual_critic", "run_logical_critic")
#     graph_builder.add_edge("run_logical_critic", "run_completeness_critic")
#     graph_builder.add_edge("run_completeness_critic", END)
#
#     return graph_builder.compile()


def build_arbitration_graph():
    graph_builder = StateGraph(ArbitrationState)

    graph_builder.add_node("dispatch", dispatch)
    graph_builder.add_node("run_factual_critic", run_factual_critic)
    graph_builder.add_node("run_logical_critic", run_logical_critic)
    graph_builder.add_node("run_completeness_critic", run_completeness_critic)
    graph_builder.add_node("collect", collect)
    graph_builder.add_node("adjudicate", adjudicate)
    graph_builder.add_node("calculate_totals", calculate_totals)

    graph_builder.add_edge(START, "dispatch")
    graph_builder.add_edge("dispatch", "run_factual_critic")
    graph_builder.add_edge("dispatch", "run_logical_critic")
    graph_builder.add_edge("dispatch", "run_completeness_critic")
    graph_builder.add_edge(
        [
            "run_factual_critic",
            "run_logical_critic",
            "run_completeness_critic",
        ],
        "collect",
    )
    graph_builder.add_edge("collect", "adjudicate")
    graph_builder.add_edge("adjudicate", "calculate_totals")
    graph_builder.add_edge("calculate_totals", END)

    return graph_builder.compile()


arbitration_graph = build_arbitration_graph()


if __name__ == "__main__":
    initial_state: ArbitrationState = {
        "input_text": (
            "The Eiffel Tower was built in 1750 and stands 1200 meters tall. "
            "All birds can fly. Penguins are birds, so penguins can fly. "
            "This proves that flying ability is universal across animals. "
            "Neural networks learn by updating weights."
        ),
        "original_prompt": "Explain how neural networks learn and give accurate examples.",
        "critiques": [],
        "verdict": {},
        "errors": [],
        "critics_available": 0,
        "graph_started_at": 0.0,
        "adjudicator_tokens_used": 0,
        "adjudicator_model_used": "",
        "adjudicator_latency_ms": 0.0,
        "arbitration_result": {},
    }

    final_state = arbitration_graph.invoke(initial_state)

    print("Final arbitration state:")
    print(final_state)

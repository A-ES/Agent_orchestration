from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph import arbitration_graph
from schemas import ArbitrationResult


TEST_CASES = [
    {
        "id": "fact-001",
        "text": (
            "The Apollo 11 mission landed on Mars in 1969. Neil Armstrong and "
            "Buzz Aldrin spent three days driving a rover across the Martian surface."
        ),
        "original_prompt": "Summarize the Apollo 11 mission.",
        "expected_quality": "bad",
        "known_issues": ["Apollo 11 landed on the Moon, not Mars", "No rover was used"],
    },
    {
        "id": "fact-002",
        "text": (
            "Marie Curie won the Nobel Prize in Literature in 1911 for writing "
            "about radioactivity. She was the first person to discover electricity."
        ),
        "original_prompt": "Explain Marie Curie's scientific achievements.",
        "expected_quality": "bad",
        "known_issues": ["Wrong Nobel category", "Did not discover electricity"],
    },
    {
        "id": "fact-003",
        "text": (
            "The Great Wall of China is about 900 kilometers long and was built "
            "entirely during the Ming dynasty to keep out Roman armies."
        ),
        "original_prompt": "Describe the Great Wall of China.",
        "expected_quality": "bad",
        "known_issues": ["Length is far too short", "Not built entirely during Ming", "Roman armies claim is wrong"],
    },
    {
        "id": "fact-004",
        "text": (
            "Python was created by James Gosling in 2005 as a replacement for Java. "
            "Its first public release included mandatory static typing."
        ),
        "original_prompt": "Give a brief history of Python.",
        "expected_quality": "bad",
        "known_issues": ["Creator is Guido van Rossum", "Initial release was 1991", "Python is dynamically typed"],
    },
    {
        "id": "fact-005",
        "text": (
            "The human heart normally has three chambers and pumps oxygen directly "
            "into the lungs through the aorta before it reaches the body."
        ),
        "original_prompt": "Explain how the human heart circulates blood.",
        "expected_quality": "bad",
        "known_issues": ["Human heart has four chambers", "Aorta carries oxygenated blood to the body"],
    },
    {
        "id": "logic-001",
        "text": (
            "All encrypted messages are secure. This message is secure. Therefore, "
            "this message must be encrypted."
        ),
        "original_prompt": "Assess whether encryption guarantees security.",
        "expected_quality": "bad",
        "known_issues": ["Affirming the consequent"],
    },
    {
        "id": "logic-002",
        "text": (
            "The product is cheaper than competitors, so it must be higher quality. "
            "Because it is higher quality, the low price proves customers value it more."
        ),
        "original_prompt": "Evaluate a product pricing argument.",
        "expected_quality": "bad",
        "known_issues": ["Non-sequitur from cheaper to higher quality", "Circular reasoning"],
    },
    {
        "id": "logic-003",
        "text": (
            "Remote work improves focus because employees avoid office interruptions. "
            "Remote work also reduces focus because there are no distractions at home."
        ),
        "original_prompt": "Discuss whether remote work improves focus.",
        "expected_quality": "bad",
        "known_issues": ["Contradicts itself about distractions and focus"],
    },
    {
        "id": "logic-004",
        "text": (
            "If a database is backed up, it can never lose data. This database lost "
            "data, so backups are useless in every organization."
        ),
        "original_prompt": "Explain the role of backups in reliability.",
        "expected_quality": "bad",
        "known_issues": ["Overgeneralization", "Invalid conclusion from one case"],
    },
    {
        "id": "logic-005",
        "text": (
            "The survey was completed by ten people from one company. Therefore, "
            "the results prove that all software engineers worldwide prefer Rust."
        ),
        "original_prompt": "Interpret a small workplace survey.",
        "expected_quality": "bad",
        "known_issues": ["Unsupported conclusion from tiny unrepresentative sample"],
    },
    {
        "id": "complete-001",
        "text": "Neural networks learn by updating weights with gradient descent.",
        "original_prompt": "Explain how neural networks learn, including loss functions, backpropagation, and examples.",
        "expected_quality": "bad",
        "known_issues": ["Missing loss functions", "Missing backpropagation", "Missing examples"],
    },
    {
        "id": "complete-002",
        "text": "HTTP/2 is faster because it allows multiple requests on one connection.",
        "original_prompt": "Explain HTTP/2 improvements over HTTP/1.1, including multiplexing, header compression, server push, and limitations.",
        "expected_quality": "bad",
        "known_issues": ["Missing header compression", "Missing server push", "Missing limitations"],
    },
    {
        "id": "complete-003",
        "text": "Photosynthesis lets plants make sugar using sunlight.",
        "original_prompt": "Explain photosynthesis, including inputs, outputs, chlorophyll, light reactions, and the Calvin cycle.",
        "expected_quality": "bad",
        "known_issues": ["Missing inputs and outputs", "Missing chlorophyll", "Missing light reactions and Calvin cycle"],
    },
    {
        "id": "complete-004",
        "text": "A REST API uses HTTP endpoints to exchange data.",
        "original_prompt": "Explain REST API design, covering resources, HTTP verbs, status codes, statelessness, pagination, and error handling.",
        "expected_quality": "bad",
        "known_issues": ["Missing verbs", "Missing status codes", "Missing statelessness", "Missing pagination and errors"],
    },
    {
        "id": "complete-005",
        "text": "Docker packages applications so they can run in containers.",
        "original_prompt": "Explain Docker containers, images, layers, registries, networking, volumes, and when not to use Docker.",
        "expected_quality": "bad",
        "known_issues": ["Missing images and layers", "Missing registries", "Missing networking and volumes", "Missing tradeoffs"],
    },
    {
        "id": "good-001",
        "text": (
            "Apollo 11 was the first crewed mission to land humans on the Moon, "
            "touching down on July 20, 1969. Neil Armstrong and Buzz Aldrin walked "
            "on the lunar surface while Michael Collins remained in lunar orbit."
        ),
        "original_prompt": "Summarize the Apollo 11 mission.",
        "expected_quality": "good",
        "known_issues": [],
    },
    {
        "id": "good-002",
        "text": (
            "Neural networks learn by comparing predictions with target values using "
            "a loss function. Backpropagation computes gradients of that loss with "
            "respect to weights, and optimizers such as gradient descent update the "
            "weights to reduce future error."
        ),
        "original_prompt": "Explain how neural networks learn, including loss functions and backpropagation.",
        "expected_quality": "good",
        "known_issues": [],
    },
    {
        "id": "good-003",
        "text": (
            "HTTP/2 improves on HTTP/1.1 by multiplexing streams over one connection, "
            "compressing headers with HPACK, and allowing server push. It reduces "
            "head-of-line blocking at the HTTP layer, though TCP-level blocking can "
            "still occur."
        ),
        "original_prompt": "Explain HTTP/2 improvements and limitations.",
        "expected_quality": "good",
        "known_issues": [],
    },
    {
        "id": "good-004",
        "text": (
            "Photosynthesis uses light energy, water, and carbon dioxide to produce "
            "glucose and oxygen. Chlorophyll absorbs light for the light reactions, "
            "which generate energy carriers that support carbon fixation in the "
            "Calvin cycle."
        ),
        "original_prompt": "Explain photosynthesis, including inputs, outputs, chlorophyll, light reactions, and the Calvin cycle.",
        "expected_quality": "good",
        "known_issues": [],
    },
    {
        "id": "good-005",
        "text": (
            "REST APIs model data as resources addressed by URLs. Clients use HTTP "
            "verbs such as GET, POST, PATCH, and DELETE, and servers communicate "
            "outcomes with status codes. Good APIs are stateless, paginate large "
            "collections, and return consistent error bodies."
        ),
        "original_prompt": "Explain REST API design, covering resources, HTTP verbs, status codes, statelessness, pagination, and error handling.",
        "expected_quality": "good",
        "known_issues": [],
    },
]


def build_initial_state(text: str, original_prompt: str) -> dict:
    return {
        "input_text": text,
        "original_prompt": original_prompt,
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


def run_arbitration_case(test_case: dict) -> ArbitrationResult:
    final_state = arbitration_graph.invoke(
        build_initial_state(
            text=test_case["text"],
            original_prompt=test_case["original_prompt"],
        )
    )
    return ArbitrationResult(**final_state["arbitration_result"])


def classify_result(result: ArbitrationResult) -> str:
    return "good" if result.verdict.quality_score >= 6 else "bad"


def calculate_rates(records: list[dict]) -> dict:
    correct = sum(record["correct"] for record in records)
    good_cases = [record for record in records if record["expected"] == "good"]
    bad_cases = [record for record in records if record["expected"] == "bad"]
    false_positives = [
        record
        for record in good_cases
        if record["predicted"] == "bad"
    ]
    false_negatives = [
        record
        for record in bad_cases
        if record["predicted"] == "good"
    ]

    return {
        "overall_accuracy": correct / len(records) * 100,
        "false_positive_rate": len(false_positives) / len(good_cases) * 100,
        "false_negative_rate": len(false_negatives) / len(bad_cases) * 100,
        "avg_cost": sum(record["cost"] for record in records) / len(records),
        "avg_latency": sum(record["latency_ms"] for record in records) / len(records),
        "false_positive_ids": [record["id"] for record in false_positives],
        "false_negative_ids": [record["id"] for record in false_negatives],
    }


def print_report(records: list[dict]) -> None:
    rates = calculate_rates(records)

    print("\n" + "=" * 90)
    print("LLM Output Arbitration Eval Report")
    print("=" * 90)
    print(f"Cases: {len(records)}")
    print(f"Overall accuracy: {rates['overall_accuracy']:.1f}%")
    print(f"False positive rate: {rates['false_positive_rate']:.1f}%")
    print(f"False negative rate: {rates['false_negative_rate']:.1f}%")
    print(f"Avg cost per arbitration: ${rates['avg_cost']:.6f}")
    print(f"Avg latency per arbitration: {rates['avg_latency']:.0f}ms")
    print(f"False positive IDs: {rates['false_positive_ids'] or 'none'}")
    print(f"False negative IDs: {rates['false_negative_ids'] or 'none'}")

    print("\nPer-case results:")
    print(
        f"{'ID':<14} {'Expected':<9} {'Predicted':<9} {'Score':<7} "
        f"{'Correct':<8} {'Cost':<11} {'Latency'}"
    )
    print("-" * 90)
    for record in records:
        print(
            f"{record['id']:<14} {record['expected']:<9} {record['predicted']:<9} "
            f"{record['score']:<7} {str(record['correct']):<8} "
            f"${record['cost']:<10.6f} {record['latency_ms']:.0f}ms"
        )


def run_eval() -> list[dict]:
    records = []

    for index, test_case in enumerate(TEST_CASES, start=1):
        print(f"\n[{index}/20] Running {test_case['id']}...")
        result = run_arbitration_case(test_case)
        predicted = classify_result(result)
        correct = predicted == test_case["expected_quality"]

        records.append(
            {
                "id": test_case["id"],
                "expected": test_case["expected_quality"],
                "predicted": predicted,
                "score": result.verdict.quality_score,
                "correct": correct,
                "cost": result.total_cost_usd,
                "latency_ms": result.total_latency_ms,
                "tokens": result.total_tokens_used,
                "known_issues": test_case["known_issues"],
                "summary": result.verdict.summary,
            }
        )

    print_report(records)
    return records


if __name__ == "__main__":
    run_eval()

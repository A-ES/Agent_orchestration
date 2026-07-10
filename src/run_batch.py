"""Batch run: 10 test inputs through full arbitration pipeline. Record metrics."""

import json
from collections import defaultdict
from time import perf_counter

from orchestrator import run_arbitration


def main():
    with open("test_inputs.json") as f:
        test_cases = json.load(f)

    results = []
    critic_issue_counts = defaultdict(int)
    critic_total_scores = defaultdict(list)

    print(f"Running batch of {len(test_cases)} test inputs...\n")

    for i, case in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] {case['name']}...", end=" ", flush=True)
        start = perf_counter()

        result = run_arbitration(text=case["text"], original_prompt=case["prompt"])

        elapsed = (perf_counter() - start) * 1000
        print(f"done ({elapsed:.0f}ms, score={result.verdict.quality_score}/10)")

        # Track per-critic stats
        for c in result.critiques:
            critic_issue_counts[c.critic_type] += len(c.issues)
            critic_total_scores[c.critic_type].append(c.score)

        results.append({
            "name": case["name"],
            "quality_score": result.verdict.quality_score,
            "confidence": result.verdict.confidence,
            "confirmed_issues": len(result.verdict.confirmed_issues),
            "dismissed_flags": len(result.verdict.dismissed_flags),
            "critics_agreed": result.verdict.critics_agreed,
            "latency_ms": result.total_latency_ms,
            "tokens": result.total_tokens_used,
            "cost_usd": result.total_cost_usd,
            "per_critic": {
                c.critic_type: {"score": c.score, "issues": len(c.issues)}
                for c in result.critiques
            },
        })

    # -----------------------------------------------------------------------
    # Summary tables
    # -----------------------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print(f"  BATCH RESULTS ({len(test_cases)} inputs)")
    print(f"{'=' * 80}")
    print(f"  {'#':<3} {'Test':<40} {'Score':<7} {'Confirmed':<11} {'Dismissed':<11} {'Latency':<10} {'Cost'}")
    print(f"  {'-'*3} {'-'*40} {'-'*6} {'-'*10} {'-'*10} {'-'*9} {'-'*10}")
    for i, r in enumerate(results, 1):
        print(
            f"  {i:<3} {r['name']:<40} {r['quality_score']}/10   "
            f"{r['confirmed_issues']:<11} {r['dismissed_flags']:<11} "
            f"{r['latency_ms']:>7,.0f}ms  ${r['cost_usd']:.5f}"
        )

    # Critic issue counts — the resume metric
    print(f"\n{'=' * 80}")
    print(f"  CRITIC ISSUE COUNTS (across {len(test_cases)} inputs)")
    print(f"{'=' * 80}")
    print(f"  {'Critic':<25} {'Total Issues':<15} {'Avg Issues/Input':<18} {'Avg Score'}")
    print(f"  {'-'*25} {'-'*14} {'-'*17} {'-'*10}")
    for critic_type in sorted(critic_issue_counts, key=critic_issue_counts.get, reverse=True):
        total = critic_issue_counts[critic_type]
        avg_issues = total / len(test_cases)
        avg_score = sum(critic_total_scores[critic_type]) / len(critic_total_scores[critic_type])
        print(f"  {critic_type:<25} {total:<15} {avg_issues:<18.1f} {avg_score:.1f}/5")

    # Aggregate metrics
    total_cost = sum(r["cost_usd"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    avg_score = sum(r["quality_score"] for r in results) / len(results)

    print(f"\n{'=' * 80}")
    print(f"  AGGREGATE METRICS")
    print(f"{'=' * 80}")
    print(f"  Total cost:        ${total_cost:.6f}")
    print(f"  Total tokens:      {total_tokens:,}")
    print(f"  Avg latency:       {avg_latency:,.0f} ms")
    print(f"  Avg quality score: {avg_score:.1f}/10")
    print(f"  Score distribution:")

    scores = [r["quality_score"] for r in results]
    low = sum(1 for s in scores if 1 <= s <= 3)
    mid = sum(1 for s in scores if 4 <= s <= 6)
    high = sum(1 for s in scores if 7 <= s <= 10)
    print(f"    Low  (1-3):  {low} inputs")
    print(f"    Mid  (4-6):  {mid} inputs")
    print(f"    High (7-10): {high} inputs")

    # Save raw results to JSON
    with open("batch_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Raw results saved to batch_results.json")


if __name__ == "__main__":
    main()

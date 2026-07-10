"""Run the full arbitration graph end-to-end on 3 test inputs. Record metrics."""

from orchestrator import run_arbitration

# ---------------------------------------------------------------------------
# Test cases: varying quality levels
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "Terrible — all wrong",
        "prompt": "Explain what Python is, how it executes code, and its concurrency model.",
        "text": (
            "Python was created by Guido van Rossum in 1995. It is a compiled language "
            "that runs directly on bare metal without any interpreter. Python is slower "
            "than C because it is compiled, but faster than C because it is interpreted. "
            "The GIL allows true multi-threading parallelism, which is why Python "
            "dominates high-frequency trading systems."
        ),
    },
    {
        "name": "Mixed — some right, some wrong",
        "prompt": "How does HTTP/2 improve on HTTP/1.1?",
        "text": (
            "HTTP/2 introduces multiplexing, allowing multiple requests over a single "
            "TCP connection. It uses header compression via HPACK to reduce overhead. "
            "However, HTTP/2 requires encryption (TLS) by specification, making "
            "unencrypted HTTP/2 impossible. Server push allows the server to send "
            "resources before the client asks. HTTP/2 was released in 2020 as RFC 9113."
        ),
    },
    {
        "name": "Good — mostly accurate, minor gaps",
        "prompt": "Explain how garbage collection works in Java.",
        "text": (
            "Java uses automatic garbage collection to manage memory. The JVM tracks "
            "object references and reclaims memory when objects are no longer reachable "
            "from GC roots. Modern JVMs use generational collection: young generation "
            "for short-lived objects (collected via minor GC) and old generation for "
            "long-lived objects (collected via major GC). The G1 collector divides the "
            "heap into regions and prioritizes collecting regions with the most garbage. "
            "GC pauses can be tuned with flags like -XX:MaxGCPauseMillis."
        ),
    },
]


def main():
    all_metrics = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n{'=' * 70}")
        print(f"  TEST {i}: {case['name']}")
        print(f"{'=' * 70}")
        print(f"  Prompt: {case['prompt']}")
        print(f"  Text:   {case['text'][:80]}...")
        print()

        result = run_arbitration(text=case["text"], original_prompt=case["prompt"])

        # Print critic results
        for c in result.critiques:
            name = c.critic_type.replace("_", " ").title()
            print(f"  [{name}] score={c.score}/5  issues={len(c.issues)}  "
                  f"confidence={c.self_confidence}/5  latency={c.latency_ms:.0f}ms  "
                  f"tokens={c.tokens_used}")

        # Print verdict
        v = result.verdict
        print(f"\n  VERDICT:")
        print(f"    Quality score:    {v.quality_score}/10")
        print(f"    Confidence:       {v.confidence}/5")
        print(f"    Critics agreed:   {v.critics_agreed}")
        print(f"    Confirmed issues: {len(v.confirmed_issues)}")
        print(f"    Dismissed flags:  {len(v.dismissed_flags)}")
        print(f"    Summary: {v.summary}")

        # Print metrics
        print(f"\n  METRICS:")
        print(f"    Total latency:  {result.total_latency_ms:,.0f} ms")
        print(f"    Total tokens:   {result.total_tokens_used:,}")
        print(f"    Total cost:     ${result.total_cost_usd:.6f}")

        all_metrics.append({
            "name": case["name"],
            "quality_score": v.quality_score,
            "confirmed_issues": len(v.confirmed_issues),
            "dismissed_flags": len(v.dismissed_flags),
            "critics_agreed": v.critics_agreed,
            "latency_ms": result.total_latency_ms,
            "tokens": result.total_tokens_used,
            "cost_usd": result.total_cost_usd,
        })

    # Summary table
    print(f"\n\n{'=' * 70}")
    print(f"  METRICS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Test':<30} {'Score':<8} {'Issues':<9} {'Dismissed':<11} {'Latency':<12} {'Tokens':<10} {'Cost'}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*10} {'-'*11} {'-'*9} {'-'*10}")
    for m in all_metrics:
        print(
            f"  {m['name']:<30} {m['quality_score']}/10   "
            f"{m['confirmed_issues']:<9} {m['dismissed_flags']:<11} "
            f"{m['latency_ms']:>8,.0f} ms  {m['tokens']:>7,}   ${m['cost_usd']:.6f}"
        )

    total_cost = sum(m["cost_usd"] for m in all_metrics)
    total_tokens = sum(m["tokens"] for m in all_metrics)
    avg_latency = sum(m["latency_ms"] for m in all_metrics) / len(all_metrics)
    print(f"\n  Totals across 3 runs:")
    print(f"    Total cost:     ${total_cost:.6f}")
    print(f"    Total tokens:   {total_tokens:,}")
    print(f"    Avg latency:    {avg_latency:,.0f} ms")


if __name__ == "__main__":
    main()

"""Benchmark sequential vs parallel execution of all 3 critics."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from critics.completeness import CompletenessCritic
from critics.factual_accuracy import FactualAccuracyCritic
from critics.logical_consistency import LogicalConsistencyCritic

BAD_TEXT = (
    "Python was created by Guido van Rossum in 1995. It is a compiled language "
    "that runs directly on bare metal without any interpreter. Python is slower "
    "than C because it is compiled, but faster than C because it is interpreted. "
    "The GIL allows true multi-threading parallelism, which is why Python "
    "dominates high-frequency trading systems."
)

ORIGINAL_PROMPT = (
    "Explain what Python is, how it executes code, its concurrency model, "
    "and its common use cases."
)


def run_sequential():
    """Run all 3 critics one after another."""
    fa = FactualAccuracyCritic()
    lc = LogicalConsistencyCritic()
    comp = CompletenessCritic()

    start = perf_counter()

    fa_result = fa.critique(BAD_TEXT)
    lc_result = lc.critique(BAD_TEXT)
    comp_result = comp.critique(text=BAD_TEXT, original_prompt=ORIGINAL_PROMPT)

    elapsed = (perf_counter() - start) * 1000
    return [fa_result, lc_result, comp_result], elapsed


def run_parallel():
    """Run all 3 critics concurrently using threads."""
    fa = FactualAccuracyCritic()
    lc = LogicalConsistencyCritic()
    comp = CompletenessCritic()

    start = perf_counter()

    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_fa = executor.submit(fa.critique, BAD_TEXT)
        fut_lc = executor.submit(lc.critique, BAD_TEXT)
        fut_comp = executor.submit(comp.critique, BAD_TEXT, ORIGINAL_PROMPT)

        fa_result = fut_fa.result()
        lc_result = fut_lc.result()
        comp_result = fut_comp.result()

    elapsed = (perf_counter() - start) * 1000
    return [fa_result, lc_result, comp_result], elapsed


def main():
    print("Running sequential benchmark...")
    seq_results, seq_time = run_sequential()

    print("Running parallel benchmark...")
    par_results, par_time = run_parallel()

    speedup = seq_time / par_time

    print(f"\n{'=' * 50}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'=' * 50}")
    print(f"  Sequential:  {seq_time:,.0f} ms")
    print(f"  Parallel:    {par_time:,.0f} ms")
    print(f"  Speedup:     {speedup:.2f}x")
    print(f"{'=' * 50}")

    # Breakdown per critic
    print(f"\n  Per-critic latency (from internal timing):")
    print(f"  {'Critic':<25} {'Sequential':<15} {'Parallel'}")
    print(f"  {'-'*25} {'-'*14} {'-'*14}")
    for s, p in zip(seq_results, par_results):
        name = s.critic_type.replace("_", " ").title()
        print(f"  {name:<25} {s.latency_ms:,.0f} ms       {p.latency_ms:,.0f} ms")

    print(f"\n  Wall-clock bottleneck (parallel) = slowest critic")
    slowest = max(par_results, key=lambda r: r.latency_ms)
    print(f"  Slowest: {slowest.critic_type} at {slowest.latency_ms:,.0f} ms")
    print(f"  Theoretical min parallel time ≈ {slowest.latency_ms:,.0f} ms")
    print(f"  Actual parallel time:            {par_time:,.0f} ms")
    print(f"  Overhead:                        {par_time - slowest.latency_ms:,.0f} ms")


if __name__ == "__main__":
    main()

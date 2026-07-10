"""Run all 3 critics on the same bad text and compare their outputs."""

from critics.completeness import CompletenessCritic
from critics.factual_accuracy import FactualAccuracyCritic
from critics.logical_consistency import LogicalConsistencyCritic

# Intentionally bad text: factually wrong, logically contradictory, and incomplete.
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


def separator(title: str) -> str:
    return f"\n{'=' * 60}\n  {title}\n{'=' * 60}"


def print_critique(result) -> None:
    print(f"  Score:           {result.score}/5")
    print(f"  Self-confidence: {result.self_confidence}/5")
    print(f"  Model:           {result.model_used}")
    print(f"  Latency:         {result.latency_ms:.0f} ms")
    print(f"  Tokens:          {result.tokens_used}")
    print(f"  Issues ({len(result.issues)}):")
    for i, issue in enumerate(result.issues, 1):
        print(f"    {i}. [{issue.severity.upper()}] {issue.description}")
        print(f"       Quote: \"{issue.quote}\"")


def main() -> None:
    print(separator("INPUT TEXT"))
    print(f"  {BAD_TEXT}\n")
    print(f"  Original prompt: {ORIGINAL_PROMPT}")

    # --- Factual Accuracy ---
    print(separator("FACTUAL ACCURACY CRITIC"))
    fa_critic = FactualAccuracyCritic()
    fa_result = fa_critic.critique(BAD_TEXT)
    print_critique(fa_result)

    # --- Logical Consistency ---
    print(separator("LOGICAL CONSISTENCY CRITIC"))
    lc_critic = LogicalConsistencyCritic()
    lc_result = lc_critic.critique(BAD_TEXT)
    print_critique(lc_result)

    # --- Completeness ---
    print(separator("COMPLETENESS CRITIC"))
    comp_critic = CompletenessCritic()
    comp_result = comp_critic.critique(text=BAD_TEXT, original_prompt=ORIGINAL_PROMPT)
    print_critique(comp_result)

    # --- Summary comparison ---
    print(separator("COMPARISON SUMMARY"))
    print(f"  {'Critic':<25} {'Score':<8} {'Issues':<8} {'Confidence':<12} {'Latency'}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*11} {'-'*10}")
    for name, r in [
        ("Factual Accuracy", fa_result),
        ("Logical Consistency", lc_result),
        ("Completeness", comp_result),
    ]:
        print(f"  {name:<25} {r.score}/5    {len(r.issues):<8} {r.self_confidence}/5        {r.latency_ms:.0f} ms")

    # Overlap analysis
    print(f"\n  Total unique issues found: {len(fa_result.issues) + len(lc_result.issues) + len(comp_result.issues)}")
    print(f"    - Factual issues:    {len(fa_result.issues)}")
    print(f"    - Logic issues:      {len(lc_result.issues)}")
    print(f"    - Completeness gaps: {len(comp_result.issues)}")


if __name__ == "__main__":
    main()

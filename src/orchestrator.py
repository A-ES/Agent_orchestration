"""Full arbitration orchestrator: parallel critics → dedup → adjudicator → verdict."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from time import perf_counter

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from critics.base import BaseCritic, CriticFailureException
from critics.completeness import CompletenessCritic
from critics.factual_accuracy import FactualAccuracyCritic
from critics.logical_consistency import LogicalConsistencyCritic
from schemas import ArbitrationResult, Critique, CritiqueIssue, DismissedFlag, Verdict


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _quotes_overlap(q1: str, q2: str, threshold: float = 0.6) -> bool:
    """Check if two issue quotes refer to the same text span."""
    return SequenceMatcher(None, q1.lower(), q2.lower()).ratio() >= threshold


def _group_overlapping_issues(
    critiques: list[Critique],
) -> list[dict]:
    """Group issues across critics that share overlapping quotes.

    Returns a list of groups, each with the issues and originating critics.
    """
    all_issues: list[tuple[str, CritiqueIssue]] = []
    for c in critiques:
        for issue in c.issues:
            all_issues.append((c.critic_type, issue))

    groups: list[list[tuple[str, CritiqueIssue]]] = []
    used = set()

    for i, (ct1, iss1) in enumerate(all_issues):
        if i in used:
            continue
        group = [(ct1, iss1)]
        used.add(i)
        for j, (ct2, iss2) in enumerate(all_issues):
            if j in used:
                continue
            if _quotes_overlap(iss1.quote, iss2.quote):
                group.append((ct2, iss2))
                used.add(j)
        groups.append(group)

    return [
        {
            "critics": list({ct for ct, _ in g}),
            "issues": [{"critic": ct, "description": iss.description, "quote": iss.quote, "severity": iss.severity} for ct, iss in g],
        }
        for g in groups
    ]


# ---------------------------------------------------------------------------
# Adjudicator
# ---------------------------------------------------------------------------

ADJUDICATOR_SYSTEM_PROMPT = (
    "You are a senior adjudicator. You receive critiques from multiple specialist "
    "critics (factual accuracy, logical consistency, completeness). Your job is to:\n"
    "1. Confirm issues that are valid.\n"
    "2. Dismiss issues that are false flags, with a reason.\n"
    "3. Produce a final quality score (1-10).\n"
    "4. Note whether the critics substantially agreed.\n"
    "5. Write a brief summary.\n\n"
    "Respond ONLY in valid JSON matching this structure:\n"
    '{"quality_score": int, "confidence": int, '
    '"confirmed_issues": [{"description": str, "quote": str, "severity": str}], '
    '"dismissed_flags": [{"issue": {"description": str, "quote": str, "severity": str}, "reason": str}], '
    '"critics_agreed": bool, "summary": str}\n\n'
    "Rules:\n"
    "- quality_score: 1-10 (10 is perfect). You MUST derive this score "
    "EXCLUSIVELY from the critic scores and issues below. Use this formula as "
    "a baseline: quality_score = round(average_critic_score * 2). Then subtract "
    "1 point for each confirmed high-severity issue, 0.5 for each medium. "
    "Do NOT penalize text for being short, informal, or lacking depth unless "
    "a critic explicitly flagged it as an issue.\n"
    "- confidence: 1-5.\n"
    "- severity: low/medium/high.\n"
    "- No markdown fences or commentary outside JSON."
)


def _run_adjudicator(
    text: str,
    critiques: list[Critique],
    issue_groups: list[dict],
    critics_agreed: bool,
) -> tuple[Verdict, int, float]:
    """Call the adjudicator LLM and return the verdict + token/latency info."""
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    scores_summary = {c.critic_type: c.score for c in critiques}
    disagreement_note = (
        "" if critics_agreed
        else "\n\n⚠️ Critics DIVERGED significantly. Resolve the tension explicitly."
    )

    user_prompt = (
        f"Original text under review:\n{text}\n\n"
        f"Critic scores: {json.dumps(scores_summary)}\n\n"
        f"Grouped issues (some may overlap across critics):\n"
        f"{json.dumps(issue_groups, indent=2)}\n"
        f"{disagreement_note}"
    )

    start = perf_counter()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": ADJUDICATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    latency_ms = (perf_counter() - start) * 1000
    tokens_used = response.usage.total_tokens if response.usage else 0
    raw = response.choices[0].message.content or ""

    payload = json.loads(raw)
    verdict = Verdict(
        quality_score=payload["quality_score"],
        confidence=payload["confidence"],
        confirmed_issues=[CritiqueIssue(**i) for i in payload.get("confirmed_issues", [])],
        dismissed_flags=[
            DismissedFlag(issue=CritiqueIssue(**d["issue"]), reason=d["reason"])
            for d in payload.get("dismissed_flags", [])
        ],
        critics_agreed=payload["critics_agreed"],
        summary=payload["summary"],
    )
    return verdict, tokens_used, latency_ms


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

TOKEN_COST_PER_TOKEN = {
    "deepseek-chat": 0.0000007,
    "deepseek-reasoner": 0.0000014,
}


def run_arbitration(text: str, original_prompt: str = "") -> ArbitrationResult:
    """Full end-to-end arbitration pipeline."""
    pipeline_start = perf_counter()

    # 1. Parallel critic dispatch
    fa = FactualAccuracyCritic()
    lc = LogicalConsistencyCritic()
    comp = CompletenessCritic()

    critiques: list[Critique] = []
    failed_critics: list[str] = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(fa.critique, text, original_prompt): "factual_accuracy",
            pool.submit(lc.critique, text, original_prompt): "logical_consistency",
            pool.submit(comp.critique, text, original_prompt): "completeness",
        }
        for fut in as_completed(futures):
            critic_name = futures[fut]
            try:
                critiques.append(fut.result())
            except CriticFailureException as exc:
                failed_critics.append(critic_name)
                print(f"  ⚠️  {critic_name} degraded: {exc}")

    # 2. Disagreement detection
    if critiques:
        scores = [c.score for c in critiques]
        critics_agreed = (max(scores) - min(scores)) <= 1
    else:
        critics_agreed = True

    # 3. Issue grouping / deduplication
    issue_groups = _group_overlapping_issues(critiques)

    # 4. Adjudicator
    verdict, adj_tokens, adj_latency = _run_adjudicator(
        text, critiques, issue_groups, critics_agreed,
    )

    # 5. Totals
    total_latency = (perf_counter() - pipeline_start) * 1000
    total_tokens = sum(c.tokens_used for c in critiques) + adj_tokens
    total_cost = (
        sum(c.tokens_used * TOKEN_COST_PER_TOKEN.get(c.model_used, 0) for c in critiques)
        + adj_tokens * TOKEN_COST_PER_TOKEN["deepseek-chat"]
    )

    return ArbitrationResult(
        input_text=text,
        critiques=critiques,
        verdict=verdict,
        total_latency_ms=total_latency,
        total_tokens_used=total_tokens,
        total_cost_usd=total_cost,
    )

import re
from typing import Any

from schemas import Critique


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def _get_field(item: Any, field_name: str) -> Any:
    if isinstance(item, dict):
        return item[field_name]
    return getattr(item, field_name)


def _issue_description(issue: Any) -> str:
    return _get_field(issue, "description")


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if len(word) > 2 and word not in STOPWORDS}


def _descriptions_overlap(first: str, second: str) -> bool:
    first_keywords = _keywords(first)
    second_keywords = _keywords(second)

    if not first_keywords or not second_keywords:
        return False

    shared_keywords = first_keywords & second_keywords
    overlap_ratio = len(shared_keywords) / min(len(first_keywords), len(second_keywords))

    return len(shared_keywords) >= 2 and overlap_ratio >= 0.4


def detect_disagreements(critiques: list[Critique]) -> list[dict]:
    disagreements = []

    if not critiques:
        return disagreements

    scores = {
        _get_field(critique, "critic_type"): _get_field(critique, "score")
        for critique in critiques
    }
    score_values = list(scores.values())

    if max(score_values) - min(score_values) >= 2:
        disagreements.append(
            {
                "type": "score_spread",
                "description": (
                    "Critic scores differ significantly: "
                    + ", ".join(
                        f"{critic_type}={score}"
                        for critic_type, score in sorted(scores.items())
                    )
                ),
                "critics_involved": sorted(scores),
            }
        )

    for critique in critiques:
        critic_type = _get_field(critique, "critic_type")
        issues = _get_field(critique, "issues")

        for issue in issues:
            description = _issue_description(issue)
            found_by_other_critic = False

            for other_critique in critiques:
                other_critic_type = _get_field(other_critique, "critic_type")
                if other_critic_type == critic_type:
                    continue

                for other_issue in _get_field(other_critique, "issues"):
                    other_description = _issue_description(other_issue)
                    if _descriptions_overlap(description, other_description):
                        found_by_other_critic = True
                        break

                if found_by_other_critic:
                    break

            if not found_by_other_critic:
                disagreements.append(
                    {
                        "type": "unique_issue",
                        "description": (
                            f"{critic_type} flagged an issue no other critic mentioned: "
                            f"{description}"
                        ),
                        "critics_involved": [critic_type],
                        "issue": issue.model_dump() if hasattr(issue, "model_dump") else issue,
                    }
                )

    return disagreements


def has_significant_disagreement(critiques: list[Critique]) -> bool:
    return bool(detect_disagreements(critiques))

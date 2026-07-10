import json

from pydantic import ValidationError

from critics.base import BaseCritic, CriticFailureException
from schemas import Critique


SYSTEM_PROMPT = (
    "You are a logic auditor. You check whether the LLM response is internally "
    "consistent AND logically follows from the question asked. A response that "
    "addresses a completely different topic than what was asked is a logical "
    "non-sequitur — flag it."
)


class LogicalConsistencyCritic(BaseCritic):
    """Critic that audits text for logical consistency issues."""

    def __init__(self) -> None:
        super().__init__(
            critic_type="logical_consistency",
            model_name="deepseek-reasoner",
        )

    def critique(self, text: str, original_prompt: str = "") -> Critique:
        if original_prompt:
            context_block = (
                f"Original question:\n{original_prompt}\n\n"
                f"LLM response:\n{text}"
            )
        else:
            context_block = f"Text to audit:\n{text}"

        user_prompt = (
            "Audit the following LLM-generated text for logical consistency "
            "and coherence with the original question.\n\n"
            "Respond ONLY in valid JSON matching this exact structure:\n"
            '{'
            '"score": int, '
            '"issues": ['
            '{"description": str, "quote": str, "severity": str}'
            "], "
            '"self_confidence": int'
            "}\n\n"
            "Rules:\n"
            "- score must be an integer from 1 to 5, where 5 is best.\n"
            "- self_confidence must be an integer from 1 to 5.\n"
            '- severity must be one of "low", "medium", or "high".\n'
            "- Do not include markdown fences, comments, or explanatory text.\n"
            "- Check reasoning structure and whether the response logically "
            "follows from the question.\n"
            "- A response on an unrelated topic is a non-sequitur — flag as "
            "high severity.\n\n"
            f"{context_block}"
        )

        response_text, tokens_used, latency_ms = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        try:
            json_start = response_text.index("{")
            payload = json.loads(response_text[json_start:])
            return Critique(
                critic_type=self.critic_type,
                score=payload["score"],
                issues=payload.get("issues", []),
                self_confidence=payload["self_confidence"],
                model_used=self.model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )
        except (
            ValueError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValidationError,
        ) as exc:
            raise CriticFailureException(
                self.critic_type,
                f"Invalid JSON critique response: {exc}",
            ) from exc


if __name__ == "__main__":
    critic = LogicalConsistencyCritic()
    result = critic.critique(
        "All birds can fly. Penguins are birds. Therefore penguins can fly. "
        "This proves that flying ability is a universal trait of all animals."
    )
    print(result.model_dump_json(indent=2))

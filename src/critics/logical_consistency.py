import json

from pydantic import ValidationError

from critics.base import BaseCritic, CriticFailureException
from schemas import Critique


SYSTEM_PROMPT = (
    "You are a logic auditor. Your job is to find contradictions, "
    "non-sequiturs, unsupported conclusions, and reasoning errors in the given "
    "text. You do not check facts — only logical structure."
)


class LogicalConsistencyCritic(BaseCritic):
    """Critic that audits text for logical consistency issues."""

    def __init__(self) -> None:
        super().__init__(
            critic_type="logical_consistency",
            model_name="deepseek-reasoner",
        )

    def critique(self, text: str) -> Critique:
        user_prompt = (
            "Audit the following LLM-generated text for logical consistency.\n\n"
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
            "- Check only reasoning structure, not real-world factual accuracy.\n\n"
            f"Text to audit:\n{text}"
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

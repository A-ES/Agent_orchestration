import json

from pydantic import ValidationError

from critics.base import BaseCritic, CriticFailureException
from schemas import Critique


SYSTEM_PROMPT = (
    "You are a factual accuracy auditor. Your job is to identify claims in the "
    "given text that are incorrect, unsupported, or hallucinated. You are "
    "thorough and skeptical. You only flag issues you are confident about."
)


class FactualAccuracyCritic(BaseCritic):
    """Critic that audits text for factual accuracy issues."""

    def __init__(self) -> None:
        super().__init__(
            critic_type="factual_accuracy",
            model_name="deepseek-chat",
        )

    def critique(self, text: str) -> Critique:
        user_prompt = (
            "Audit the following LLM-generated text for factual accuracy.\n\n"
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
            "- Do not include markdown fences, comments, or explanatory text.\n\n"
            f"Text to audit:\n{text}"
        )

        response_text, tokens_used, latency_ms = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        try:
            payload = json.loads(response_text)
            return Critique(
                critic_type=self.critic_type,
                score=payload["score"],
                issues=payload.get("issues", []),
                self_confidence=payload["self_confidence"],
                model_used=self.model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
            raise CriticFailureException(
                self.critic_type,
                f"Invalid JSON critique response: {exc}",
            ) from exc


if __name__ == "__main__":
    critic = FactualAccuracyCritic()
    result = critic.critique(
        "The Eiffel Tower was built in 1750 and stands 1200 meters tall. "
        "It was designed by Gustave Eiffel as a permanent structure."
    )
    print(result.model_dump_json(indent=2))

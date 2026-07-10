import json

from pydantic import ValidationError

from critics.base import BaseCritic, CriticFailureException
from schemas import Critique


SYSTEM_PROMPT = (
    "You are a factual accuracy auditor. You evaluate whether an LLM-generated "
    "response is factually correct AND relevant to what was actually asked. If "
    "the response completely ignores the question and answers something else "
    "entirely, that is a factual/relevance failure — flag it as high severity."
)


class FactualAccuracyCritic(BaseCritic):
    """Critic that audits text for factual accuracy issues."""

    def __init__(self) -> None:
        super().__init__(
            critic_type="factual_accuracy",
            model_name="deepseek-chat",
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
            "Audit the following LLM-generated text for factual accuracy and "
            "relevance to the original question.\n\n"
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
            "- If the response does not address the question at all, flag it "
            "as a high-severity relevance failure.\n"
            "- Do not include markdown fences, comments, or explanatory text.\n\n"
            f"{context_block}"
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

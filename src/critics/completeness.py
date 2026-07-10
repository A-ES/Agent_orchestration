import json

from pydantic import ValidationError

from critics.base import BaseCritic, CriticFailureException
from schemas import Critique


SYSTEM_PROMPT = (
    "You are a completeness auditor. Given a question and an answer, identify "
    "what the answer failed to address, glossed over, or left ambiguous. If no "
    "original question is provided, assess whether the text seems self-complete."
)


class CompletenessCritic(BaseCritic):
    """Critic that audits whether an answer fully addresses its prompt."""

    def __init__(self) -> None:
        super().__init__(
            critic_type="completeness",
            model_name="deepseek-chat",
        )

    def critique(self, text: str, original_prompt: str = "") -> Critique:
        prompt_context = (
            original_prompt
            if original_prompt
            else "No original question was provided. Assess self-completeness."
        )
        user_prompt = (
            "Audit the following answer for completeness against the original "
            "question when available.\n\n"
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
            f"Original question:\n{prompt_context}\n\n"
            f"Answer to audit:\n{text}"
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
    original_prompt = "Explain how neural networks learn"
    text = "Neural networks use gradient descent to update weights."

    critic = CompletenessCritic()
    result = critic.critique(text=text, original_prompt=original_prompt)
    print(result.model_dump_json(indent=2))

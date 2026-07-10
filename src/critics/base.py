from abc import ABC, abstractmethod
from time import perf_counter
from typing import Literal

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from schemas import Critique


CriticType = Literal[
    "factual_accuracy",
    "logical_consistency",
    "completeness",
]


class CriticFailureException(Exception):
    """Raised when a critic cannot complete its LLM-backed critique."""

    def __init__(self, critic_type: str, message: str) -> None:
        self.critic_type = critic_type
        super().__init__(f"{critic_type} critic failed: {message}")


class BaseCritic(ABC):
    """Shared interface and utilities for all critic agents."""

    _MODEL_TOKEN_COSTS_USD = {
        "deepseek-chat": 0.0000007,
        "deepseek-reasoner": 0.0000014,
    }

    def __init__(
        self,
        critic_type: CriticType,
        model_name: str = "deepseek-chat",
    ) -> None:
        self.critic_type = critic_type
        self._model_name = model_name
        self._client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @abstractmethod
    def critique(self, text: str) -> Critique:
        """Evaluate LLM-generated text and return a structured critique."""

    @staticmethod
    def _calculate_cost(tokens: int, model: str) -> float:
        cost_per_token = BaseCritic._MODEL_TOKEN_COSTS_USD.get(model, 0.0)
        return tokens * cost_per_token

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, float]:
        start_time = perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            raise CriticFailureException(self.critic_type, str(exc)) from exc

        latency_ms = (perf_counter() - start_time) * 1000
        response_text = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0

        return response_text, tokens_used, latency_ms

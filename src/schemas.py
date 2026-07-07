from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CritiqueIssue(BaseModel):
    description: str = Field(
        ...,
        description="A concise explanation of the problem the critic identified.",
    )
    quote: str = Field(
        ...,
        description="The exact portion of the LLM output that triggered the issue.",
    )
    severity: Literal["low", "medium", "high"] = Field(
        ...,
        description="Impact level of the issue on the output's overall quality.",
    )


class Critique(BaseModel):
    critic_type: Literal[
        "factual_accuracy",
        "logical_consistency",
        "completeness",
    ] = Field(
        ...,
        description="The specialized critic role that produced this critique.",
    )
    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Critic's quality score for its dimension, where 5 is best.",
    )
    issues: list[CritiqueIssue] = Field(
        default_factory=list,
        description="Specific issues found by this critic.",
    )
    self_confidence: int = Field(
        ...,
        ge=1,
        le=5,
        description="How confident the critic is in its own judgment.",
    )
    model_used: str = Field(
        ...,
        description="Name of the LLM used to produce this critique.",
    )
    latency_ms: float = Field(
        ...,
        ge=0,
        description="Time taken by this critic in milliseconds.",
    )
    tokens_used: int = Field(
        ...,
        ge=0,
        description="Token count consumed by this critic.",
    )


class DismissedFlag(BaseModel):
    issue: CritiqueIssue = Field(
        ...,
        description="The critic issue that the adjudicator dismissed.",
    )
    reason: str = Field(
        ...,
        description="Why the adjudicator chose not to confirm this issue.",
    )


class Verdict(BaseModel):
    quality_score: int = Field(
        ...,
        description="Adjudicator's final quality score for the output, from 1 to 10.",
    )
    confidence: int = Field(
        ...,
        ge=1,
        le=5,
        description="Adjudicator confidence in the final verdict.",
    )
    confirmed_issues: list[CritiqueIssue] = Field(
        default_factory=list,
        description="Issues the adjudicator accepted as valid.",
    )
    dismissed_flags: list[DismissedFlag] = Field(
        default_factory=list,
        description="Critic-raised issues the adjudicator rejected, with reasons.",
    )
    critics_agreed: bool = Field(
        ...,
        description="Whether the critics substantially agreed with each other.",
    )
    summary: str = Field(
        ...,
        description="Short natural-language synthesis of the final judgment.",
    )

    @field_validator("quality_score")
    @classmethod
    def validate_quality_score(cls, value: int) -> int:
        if not 1 <= value <= 10:
            raise ValueError("quality_score must be between 1 and 10")
        return value


class ArbitrationResult(BaseModel):
    input_text: str = Field(
        ...,
        description="The original LLM-generated text submitted for arbitration.",
    )
    critiques: list[Critique] = Field(
        ...,
        description="Critiques returned by the specialized critic agents.",
    )
    verdict: Verdict = Field(
        ...,
        description="Final adjudicator judgment synthesized from the critiques.",
    )
    total_latency_ms: float = Field(
        ...,
        ge=0,
        description="End-to-end arbitration latency in milliseconds.",
    )
    total_tokens_used: int = Field(
        ...,
        ge=0,
        description="Total tokens consumed across critics and adjudicator.",
    )
    total_cost_usd: float = Field(
        ...,
        ge=0,
        description="Estimated total API cost in US dollars.",
    )

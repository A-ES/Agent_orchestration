import json
from time import perf_counter

from openai import OpenAI
from pydantic import ValidationError

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from schemas import Critique, Verdict


SYSTEM_PROMPT = (
    "You are an impartial adjudicator reviewing critiques of an AI-generated "
    "text. You receive the original text, structured critiques from multiple "
    "reviewers, and a list of disagreements between them. Your job is to reason "
    "through each disagreement, decide which critic is correct with evidence, "
    "and produce a final verdict. Be specific — reference the actual text when "
    "confirming or dismissing an issue."
)


class AdjudicatorFailureException(Exception):
    """Raised when the adjudicator cannot produce a valid verdict."""


class Adjudicator:
    """Final reviewer that synthesizes critic outputs into a verdict."""

    def __init__(self, model_name: str = "deepseek-reasoner") -> None:
        self.model_name = model_name
        self.last_tokens_used = 0
        self.last_latency_ms = 0.0
        self._client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

    def adjudicate(
        self,
        input_text: str,
        critiques: list[Critique],
        disagreements: list[dict],
    ) -> Verdict:
        user_prompt = self._build_user_prompt(input_text, critiques, disagreements)

        start_time = perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            raise AdjudicatorFailureException(str(exc)) from exc

        latency_ms = (perf_counter() - start_time) * 1000
        self.last_latency_ms = latency_ms
        self.last_tokens_used = response.usage.total_tokens if response.usage else 0
        response_text = response.choices[0].message.content or ""

        try:
            json_start = response_text.index("{")
            payload = json.loads(response_text[json_start:])
            verdict = Verdict(**payload)
        except (
            ValueError,
            json.JSONDecodeError,
            TypeError,
            ValidationError,
        ) as exc:
            raise AdjudicatorFailureException(
                f"Invalid JSON verdict response: {exc}"
            ) from exc

        print(f"Completed adjudication in {latency_ms:.0f} ms.")
        print(verdict.model_dump_json(indent=2))

        return verdict

    def _build_user_prompt(
        self,
        input_text: str,
        critiques: list[Critique],
        disagreements: list[dict],
    ) -> str:
        critiques_section = "\n\n".join(
            self._format_critique(critique) for critique in critiques
        )
        disagreements_section = (
            "\n".join(
                f"- Type: {disagreement['type']}\n"
                f"  Description: {disagreement['description']}\n"
                f"  Critics involved: {', '.join(disagreement['critics_involved'])}\n"
                f"  Full data: {json.dumps(disagreement, ensure_ascii=True)}"
                for disagreement in disagreements
            )
            if disagreements
            else "No disagreements detected."
        )

        return (
            "Review the original AI-generated text, the critic reports, and the "
            "detected disagreements. Resolve disagreements using evidence from "
            "the text and the critiques.\n\n"
            "Respond ONLY in valid JSON matching this exact Verdict schema:\n"
            "{\n"
            '  "quality_score": int,\n'
            '  "confidence": int,\n'
            '  "confirmed_issues": ['
            '{"description": str, "quote": str, "severity": str}],\n'
            '  "dismissed_flags": ['
            '{"issue": {"description": str, "quote": str, "severity": str}, '
            '"reason": str}],\n'
            '  "critics_agreed": bool,\n'
            '  "summary": str\n'
            "}\n\n"
            "Rules:\n"
            "- quality_score must be an integer from 1 to 10.\n"
            "- confidence must be an integer from 1 to 5.\n"
            '- severity must be one of "low", "medium", or "high".\n'
            "- confirmed_issues should include issues you believe are valid.\n"
            "- dismissed_flags should explain rejected critic flags with evidence.\n"
            "- Do not include markdown fences, comments, or explanatory text.\n\n"
            f"Original AI-generated text:\n{input_text}\n\n"
            f"Critiques:\n{critiques_section}\n\n"
            f"Disagreements:\n{disagreements_section}"
        )

    def _format_critique(self, critique: Critique) -> str:
        issues = (
            "\n".join(
                f"  {index}. [{issue.severity}] {issue.description}\n"
                f"     Quote: {issue.quote}"
                for index, issue in enumerate(critique.issues, start=1)
            )
            if critique.issues
            else "  No issues reported."
        )

        return (
            f"Critic: {critique.critic_type}\n"
            f"Score: {critique.score}/5\n"
            f"Self-confidence: {critique.self_confidence}/5\n"
            f"Model used: {critique.model_used}\n"
            f"Issues:\n{issues}"
        )

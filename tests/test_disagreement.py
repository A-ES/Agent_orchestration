import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disagreement import detect_disagreements, has_significant_disagreement
from schemas import Critique, CritiqueIssue


class DisagreementDetectionTest(unittest.TestCase):
    def test_detects_score_spread_and_unique_issue(self) -> None:
        critiques = [
            Critique(
                critic_type="factual_accuracy",
                score=5,
                issues=[
                    CritiqueIssue(
                        description="Wrong Eiffel Tower construction year",
                        quote="built in 1750",
                        severity="high",
                    )
                ],
                self_confidence=5,
                model_used="deepseek-chat",
                latency_ms=100.0,
                tokens_used=100,
            ),
            Critique(
                critic_type="logical_consistency",
                score=3,
                issues=[
                    CritiqueIssue(
                        description="Incorrect construction year for Eiffel Tower",
                        quote="built in 1750",
                        severity="medium",
                    )
                ],
                self_confidence=4,
                model_used="deepseek-reasoner",
                latency_ms=150.0,
                tokens_used=120,
            ),
            Critique(
                critic_type="completeness",
                score=4,
                issues=[
                    CritiqueIssue(
                        description="Missing explanation of training data requirements",
                        quote="Neural networks update weights.",
                        severity="medium",
                    )
                ],
                self_confidence=4,
                model_used="deepseek-chat",
                latency_ms=90.0,
                tokens_used=80,
            ),
        ]

        disagreements = detect_disagreements(critiques)

        self.assertEqual(len(disagreements), 2)
        self.assertEqual(disagreements[0]["type"], "score_spread")
        self.assertEqual(disagreements[1]["type"], "unique_issue")
        self.assertEqual(disagreements[1]["critics_involved"], ["completeness"])
        self.assertTrue(has_significant_disagreement(critiques))


if __name__ == "__main__":
    unittest.main()

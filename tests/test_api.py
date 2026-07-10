import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api import app
from schemas import ArbitrationResult, Verdict


def fake_result(text: str) -> ArbitrationResult:
    return ArbitrationResult(
        input_text=text,
        critiques=[],
        verdict=Verdict(
            quality_score=8,
            confidence=4,
            confirmed_issues=[],
            dismissed_flags=[],
            critics_agreed=True,
            summary="Looks good.",
        ),
        total_latency_ms=123.0,
        total_tokens_used=456,
        total_cost_usd=0.00123,
    )


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("deepseek-chat", response.json()["models_available"])

    def test_arbitrate_success(self) -> None:
        with patch("api.run_arbitration") as run_arbitration:
            run_arbitration.return_value = (
                fake_result("hello"),
                {"critics_available": 3},
            )

            response = self.client.post("/v1/arbitrate", json={"text": "hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["verdict"]["quality_score"], 8)

    def test_arbitrate_rejects_empty_text(self) -> None:
        response = self.client.post("/v1/arbitrate", json={"text": "   "})

        self.assertEqual(response.status_code, 422)

    def test_arbitrate_returns_503_when_all_critics_fail(self) -> None:
        with patch("api.run_arbitration") as run_arbitration:
            run_arbitration.return_value = (
                fake_result("hello"),
                {"critics_available": 0},
            )

            response = self.client.post("/v1/arbitrate", json={"text": "hello"})

        self.assertEqual(response.status_code, 503)

    def test_batch_rejects_more_than_twenty_inputs(self) -> None:
        response = self.client.post(
            "/v1/arbitrate/batch",
            json={
                "inputs": [
                    {"id": str(index), "text": "hello"}
                    for index in range(21)
                ]
            },
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

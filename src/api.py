import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from graph import arbitration_graph
from schemas import ArbitrationResult


logger = logging.getLogger("arbitration_api")
logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title="LLM Output Arbitration System API",
    version="1.0.0",
)


class ArbitrationRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    original_prompt: str = ""

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must be non-empty")
        return stripped


class BatchArbitrationInput(ArbitrationRequest):
    id: str


class BatchArbitrationRequest(BaseModel):
    inputs: list[BatchArbitrationInput] = Field(..., min_length=1, max_length=20)


class BatchArbitrationResponse(BaseModel):
    results: list[ArbitrationResult]
    summary: dict[str, Any]


def build_initial_state(text: str, original_prompt: str = "") -> dict[str, Any]:
    return {
        "input_text": text,
        "original_prompt": original_prompt,
        "critiques": [],
        "verdict": {},
        "errors": [],
        "critics_available": 0,
        "graph_started_at": 0.0,
        "adjudicator_tokens_used": 0,
        "adjudicator_model_used": "",
        "adjudicator_latency_ms": 0.0,
        "arbitration_result": {},
    }


def run_arbitration(text: str, original_prompt: str = "") -> tuple[ArbitrationResult, dict]:
    final_state = arbitration_graph.invoke(
        build_initial_state(text=text, original_prompt=original_prompt)
    )
    result = ArbitrationResult(**final_state["arbitration_result"])
    return result, final_state


def ensure_critics_available(final_state: dict) -> None:
    if final_state.get("critics_available", 0) == 0:
        raise HTTPException(
            status_code=503,
            detail="All critics failed — arbitration unavailable",
        )


def get_input_length_from_body(body: bytes) -> int:
    if not body:
        return 0

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return 0

    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        return len(payload["text"])

    if isinstance(payload, dict) and isinstance(payload.get("inputs"), list):
        return sum(
            len(item.get("text", ""))
            for item in payload["inputs"]
            if isinstance(item, dict)
        )

    return 0


def response_log_fields(response_body: bytes) -> tuple[Any, Any, Any]:
    if not response_body:
        return None, None, None

    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError:
        return None, None, None

    if isinstance(payload, dict) and "verdict" in payload:
        return (
            payload["verdict"].get("quality_score"),
            payload.get("total_cost_usd"),
            payload.get("total_latency_ms"),
        )

    if isinstance(payload, dict) and "summary" in payload:
        summary = payload["summary"]
        return (
            summary.get("average_quality_score"),
            summary.get("total_cost_usd"),
            summary.get("total_latency_ms"),
        )

    return None, None, None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started_at = datetime.now(timezone.utc)
    request_body = await request.body()
    input_length = get_input_length_from_body(request_body)

    response = await call_next(request)
    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    verdict_score, total_cost, latency_ms = response_log_fields(response_body)
    logger.info(
        "timestamp=%s endpoint=%s input_length=%s verdict_score=%s total_cost=%s latency_ms=%s",
        started_at.isoformat(),
        request.url.path,
        input_length,
        verdict_score,
        total_cost,
        latency_ms,
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.get("/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_available": ["deepseek-chat", "deepseek-reasoner"],
    }


@app.post("/v1/arbitrate", response_model=ArbitrationResult)
def arbitrate(request: ArbitrationRequest) -> ArbitrationResult:
    result, final_state = run_arbitration(
        text=request.text,
        original_prompt=request.original_prompt,
    )
    ensure_critics_available(final_state)
    return result


@app.post("/v1/arbitrate/batch", response_model=BatchArbitrationResponse)
def arbitrate_batch(request: BatchArbitrationRequest) -> BatchArbitrationResponse:
    results = []
    failed_items = []

    for item in request.inputs:
        result, final_state = run_arbitration(
            text=item.text,
            original_prompt=item.original_prompt,
        )
        results.append(result)

        if final_state.get("critics_available", 0) == 0:
            failed_items.append(item.id)

    if len(failed_items) == len(request.inputs):
        raise HTTPException(
            status_code=503,
            detail="All critics failed for every batch input — arbitration unavailable",
        )

    total_tokens = sum(result.total_tokens_used for result in results)
    total_cost = sum(result.total_cost_usd for result in results)
    total_latency = sum(result.total_latency_ms for result in results)
    average_quality = (
        sum(result.verdict.quality_score for result in results) / len(results)
        if results
        else 0
    )

    return BatchArbitrationResponse(
        results=results,
        summary={
            "count": len(results),
            "failed_count": len(failed_items),
            "failed_ids": failed_items,
            "total_tokens_used": total_tokens,
            "total_cost_usd": total_cost,
            "total_latency_ms": total_latency,
            "average_quality_score": average_quality,
        },
    )

"""FastAPI backend for the LLM Output Arbitration System."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from orchestrator import run_arbitration
from schemas import ArbitrationResult


app = FastAPI(title="LLM Output Arbitration", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ArbitrationRequest(BaseModel):
    text: str
    original_prompt: str = ""


@app.post("/v1/arbitrate", response_model=ArbitrationResult)
def arbitrate(req: ArbitrationRequest):
    return run_arbitration(text=req.text, original_prompt=req.original_prompt)

# LLM Output Arbitration System

Multi-agent pipeline that evaluates LLM-generated text for quality using parallel critic agents and a final adjudicator.

## What It Does

- Runs three specialized critics simultaneously against any LLM output — checking facts, logic, and completeness independently.
- Detects when critics disagree and routes contested issues to an adjudicator that confirms or dismisses each flag.
- Returns a single structured verdict with a quality score, confirmed issues, and a cost/latency breakdown.

## Architecture

```
                         ┌─────────────────────┐
                         │     Input Text       │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
          ┌─────────────┐ ┌─────────────────┐ ┌──────────────┐
          │   Factual   │ │    Logical      │ │ Completeness │
          │  Accuracy   │ │  Consistency    │ │    Critic    │
          │   Critic    │ │    Critic       │ │              │
          └──────┬──────┘ └───────┬─────────┘ └──────┬───────┘
                 │                │                   │
                 └───────────────┼───────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  Disagreement Detection │
                    │  + Issue Deduplication  │
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐
                    │      Adjudicator       │
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐
                    │   Structured Verdict   │
                    │  (score, issues, cost) │
                    └────────────────────────┘
```

Critics execute in parallel via thread pool. Wall-clock latency equals the slowest critic plus adjudicator time.

## Eval Results

```
90% accuracy on 20-case eval set | $0.0046 avg cost per arbitration | 22s avg latency
```

- Zero false positives (never flags good text as bad)
- 20% false negative rate concentrated in completeness edge cases
- Factual and logical critics: 100% classification accuracy

## Tech Stack

| Component | Role |
|-----------|------|
| LangGraph | Agent orchestration and state graph |
| FastAPI | REST API layer |
| Pydantic | Schema validation for critiques and verdicts |
| OpenAI-compatible API | LLM inference (DeepSeek Chat + Reasoner) |
| Streamlit | Evaluation dashboard |
| Docker | Containerized deployment |



## Key Design Decisions

**Why three separate critics instead of one prompt?**

A single "evaluate this text" prompt conflates factual verification with logical analysis with completeness checking. These are fundamentally different cognitive tasks. Separating them lets each critic use a tailored system prompt and model — the logic critic uses a reasoning model (`deepseek-reasoner`) while the others use a faster chat model. Specialization produces more precise issue descriptions and fewer missed problems than a generalist pass.

**Why parallel dispatch?**

Critics are independent — none needs another's output. Running them sequentially would triple latency for zero benefit. Parallel execution means total critic time equals the slowest critic (~10s for the reasoner) rather than the sum of all three (~18s). The overhead of thread coordination is negligible (<5ms).

**What does the adjudicator do that self-evaluation can't?**

Self-evaluation asks the same model to judge its own output — it shares the same blind spots that produced the errors. The adjudicator receives structured critiques from multiple independent perspectives, with explicit issue quotes and severity ratings. Its job is narrower and more tractable: resolve disagreements between critics, dismiss false flags, and synthesize a final score. It sees the evidence laid out rather than re-reading raw text with the same biases.

**Why quote-based deduplication before adjudication?**

Multiple critics often flag the same text span from different angles (e.g., a factually wrong claim that also breaks logical consistency). Grouping overlapping quotes before adjudication reduces token cost and gives the adjudicator a clearer picture: "two critics flagged this sentence" is more informative than presenting the same quote twice without context.

**Graceful degradation over hard failure.**

If a critic times out or returns unparseable JSON, the pipeline continues with the remaining critics rather than failing the entire arbitration. The adjudicator notes which perspectives are missing and adjusts confidence accordingly. A two-critic verdict is better than no verdict.

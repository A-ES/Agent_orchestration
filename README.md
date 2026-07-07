# LLM Output Arbitration System

Skeleton for a Python system that will evaluate LLM-generated text using specialized critic agents and an adjudicator.

## Project Layout

```text
.
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   └── llm_output_arbitration/
│       ├── __init__.py
│       ├── config.py
│       ├── agents/
│       │   └── __init__.py
│       ├── api/
│       │   └── __init__.py
│       ├── schemas/
│       │   └── __init__.py
│       └── ui/
│           └── __init__.py
└── tests/
    └── __init__.py
```

## What Each Part Is For

- `requirements.txt`: Python dependencies for API serving, LLM calls, graph orchestration, environment loading, and Streamlit UI work.
- `.env.example`: Template for local environment variables. Copy it to `.env` and set `DEEPSEEK_API_KEY`.
- `src/`: Standard source-layout directory that keeps application code separate from project metadata and tests.
- `src/llm_output_arbitration/`: Main Python package for the arbitration system.
- `src/llm_output_arbitration/config.py`: Central place for loading environment variables and DeepSeek API settings.
- `src/llm_output_arbitration/agents/`: Placeholder package for future critic and adjudicator agents.
- `src/llm_output_arbitration/api/`: Placeholder package for future FastAPI routes and server code.
- `src/llm_output_arbitration/schemas/`: Placeholder package for future Pydantic request, response, critique, and verdict models.
- `src/llm_output_arbitration/ui/`: Placeholder package for future Streamlit UI code.
- `tests/`: Placeholder test package for future unit and integration tests.

No agent logic is implemented yet.
# Agent_orchestration

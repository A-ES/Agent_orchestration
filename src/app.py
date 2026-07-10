import streamlit as st

from disagreement import detect_disagreements
from graph import arbitration_graph
from schemas import Critique


CRITIC_LABELS = {
    "factual_accuracy": "Factual Accuracy",
    "logical_consistency": "Logical Consistency",
    "completeness": "Completeness",
}


def build_initial_state(input_text: str, original_prompt: str) -> dict:
    return {
        "input_text": input_text,
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


def score_color(score: int) -> str:
    if score < 4:
        return "#dc2626"
    if score <= 6:
        return "#ca8a04"
    return "#16a34a"


def severity_color(severity: str) -> str:
    return {
        "low": "#2563eb",
        "medium": "#ca8a04",
        "high": "#dc2626",
    }.get(severity, "#4b5563")


def render_quality_score(score: int) -> None:
    st.markdown(
        f"""
        <div class="quality-score" style="border-color: {score_color(score)};">
            <div class="metric-label">Final Quality Score</div>
            <div class="metric-value" style="color: {score_color(score)};">{score}/10</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_critic_columns(critiques: list[Critique]) -> None:
    critique_by_type = {critique.critic_type: critique for critique in critiques}
    columns = st.columns(3)

    for column, critic_type in zip(columns, CRITIC_LABELS):
        critique = critique_by_type.get(critic_type)
        with column:
            st.subheader(CRITIC_LABELS[critic_type])
            if not critique:
                st.warning("Unavailable")
                continue

            st.metric("Score", f"{critique.score}/5")
            st.caption(f"Issues found: {len(critique.issues)}")
            for issue in critique.issues:
                st.markdown(
                    f"""
                    <div class="issue-card">
                        <span class="badge" style="background: {severity_color(issue.severity)};">
                            {issue.severity.upper()}
                        </span>
                        <div class="issue-description">{issue.description}</div>
                        <div class="issue-quote">"{issue.quote}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_disagreements(disagreements: list[dict]) -> None:
    st.subheader("Disagreements")
    if not disagreements:
        st.success("No significant disagreements detected.")
        return

    for disagreement in disagreements:
        st.markdown(
            f"""
            <div class="disagreement">
                <strong>{disagreement["type"].replace("_", " ").title()}</strong>
                <p>{disagreement["description"]}</p>
                <span>{", ".join(disagreement["critics_involved"])}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_confirmed_issues(confirmed_issues: list[dict]) -> None:
    st.subheader("Confirmed Issues")
    if not confirmed_issues:
        st.info("No confirmed issues.")
        return

    for index, issue in enumerate(confirmed_issues, start=1):
        with st.expander(f"Issue {index}: {issue['description']}"):
            st.markdown(
                f"""
                <span class="badge" style="background: {severity_color(issue["severity"])};">
                    {issue["severity"].upper()}
                </span>
                """,
                unsafe_allow_html=True,
            )
            st.write(issue["quote"])


def render_results(final_state: dict) -> None:
    result = final_state["arbitration_result"]
    critiques = [Critique(**critique) for critique in result["critiques"]]
    verdict = result["verdict"]
    disagreements = detect_disagreements(critiques)

    st.divider()
    render_quality_score(verdict["quality_score"])
    st.write(verdict["summary"])

    render_critic_columns(critiques)
    render_disagreements(disagreements)
    render_confirmed_issues(verdict["confirmed_issues"])

    if final_state["errors"]:
        with st.expander("Errors"):
            for error in final_state["errors"]:
                st.error(f"{error['critic_type']}: {error['error']}")

    st.markdown(
        f"""
        <div class="footer">
            Tokens used: {result["total_tokens_used"]} |
            Cost: ${result["total_cost_usd"]:.6f} |
            Time: {result["total_latency_ms"]:.0f}ms
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="LLM Output Arbitration System", layout="wide")

st.markdown(
    """
    <style>
        .quality-score {
            border-left: 8px solid;
            padding: 1rem 1.25rem;
            background: #f8fafc;
            margin: 1rem 0 1.5rem;
        }
        .metric-label {
            color: #475569;
            font-size: 0.95rem;
            font-weight: 600;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .issue-card {
            border: 1px solid #e5e7eb;
            padding: 0.75rem;
            margin: 0.6rem 0;
            background: #ffffff;
        }
        .badge {
            color: white;
            border-radius: 4px;
            padding: 0.15rem 0.4rem;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .issue-description {
            margin-top: 0.5rem;
            font-weight: 600;
        }
        .issue-quote {
            margin-top: 0.35rem;
            color: #475569;
            font-size: 0.9rem;
        }
        .disagreement {
            background: #fff7ed;
            border-left: 6px solid #f97316;
            padding: 0.85rem 1rem;
            margin: 0.7rem 0;
        }
        .disagreement p {
            margin: 0.35rem 0;
        }
        .disagreement span {
            color: #9a3412;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .footer {
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
            color: #475569;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LLM Output Arbitration System")

original_prompt = st.text_input("Original prompt (optional)")
input_text = st.text_area("Paste LLM output to arbitrate", height=260)

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if st.button("Run Arbitration", type="primary"):
    if not input_text.strip():
        st.error("Paste LLM output to arbitrate.")
    else:
        with st.spinner("Running 3 critic agents in parallel..."):
            st.session_state.last_result = arbitration_graph.invoke(
                build_initial_state(
                    input_text=input_text.strip(),
                    original_prompt=original_prompt.strip(),
                )
            )

if st.session_state.last_result:
    render_results(st.session_state.last_result)

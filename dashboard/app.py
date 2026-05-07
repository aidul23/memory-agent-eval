"""Streamlit dashboard for browsing experiment results.

Run with:
    streamlit run dashboard/app.py

The dashboard does not require any of the heavy LLM SDKs - it only reads the
JSONL logs produced by the experiment runner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make ``src`` importable when running from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.aggregate_results import load_runs  # noqa: E402

st.set_page_config(page_title="DFx Memory Agent Dashboard", layout="wide")

st.title("DFx Memory-Agent Evaluation Dashboard")
st.caption(
    "Read-only view of experiment outputs. Point it at the folder produced "
    "by `python -m src.main run --config configs/experiment.yaml`."
)

default_results = ROOT / "results" / "raw_logs"
results_dir = st.sidebar.text_input("Results directory", value=str(default_results))


@st.cache_data(show_spinner=False)
def _load(path: str) -> pd.DataFrame:
    return load_runs(path)


df = _load(results_dir)
if df.empty:
    st.warning("No JSONL records found. Run an experiment first.")
    st.stop()


# ---- Sidebar filters ------------------------------------------------------

experiments = sorted(df["experiment_id"].dropna().unique().tolist())
sel_experiments = st.sidebar.multiselect("Experiment", experiments, default=experiments)
mem_systems = sorted(df["memory_system"].dropna().unique().tolist())
sel_memories = st.sidebar.multiselect("Memory system", mem_systems, default=mem_systems)
providers = sorted(df["llm_provider"].dropna().unique().tolist())
sel_providers = st.sidebar.multiselect("LLM provider", providers, default=providers)

f = df[
    df["experiment_id"].isin(sel_experiments)
    & df["memory_system"].isin(sel_memories)
    & df["llm_provider"].isin(sel_providers)
]

if f.empty:
    st.warning("No data after filtering.")
    st.stop()


# ---- Top-level KPIs -------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Interactions", f"{len(f)}")
c2.metric("Mean compliance", f"{f['rule_compliance_score'].mean():.2f}")
c3.metric("Success rate", f"{f['task_success'].mean():.0%}")
c4.metric("Mean latency", f"{f['latency_s'].mean():.2f}s")


# ---- Tables --------------------------------------------------------------

st.subheader("Condition summary")
condition = (
    f.groupby(["memory_system", "llm_provider", "model_name"], dropna=False)
    .agg(
        n=("task_success", "size"),
        success_rate=("task_success", "mean"),
        mean_compliance=("rule_compliance_score", "mean"),
        std_compliance=("rule_compliance_score", "std"),
        mean_relevance=("retrieval_relevance", "mean"),
        mean_correctness=("retrieval_correctness", "mean"),
        mean_usage=("usage_quality", "mean"),
        mean_latency=("latency_s", "mean"),
        mean_cost=("estimated_cost_usd", "mean"),
    )
    .reset_index()
)
st.dataframe(condition, width='stretch')


# ---- Charts --------------------------------------------------------------

st.subheader("Charts")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("**Rule compliance by memory system**")
    chart = f.groupby("memory_system")["rule_compliance_score"].mean().sort_values(ascending=False)
    st.bar_chart(chart)
with chart_col2:
    st.markdown("**Compliance over iterations**")
    pivot = (
        f.groupby(["session_id", "memory_system"])["rule_compliance_score"]
        .mean()
        .unstack("memory_system")
        .sort_index()
    )
    st.line_chart(pivot)


# ---- Failure modes -------------------------------------------------------

st.subheader("Failure modes")
fail = f[f["task_success"] == False]  # noqa: E712
if fail.empty:
    st.success("No failed interactions in the current filter.")
else:
    st.dataframe(
        fail[
            [
                "experiment_id", "memory_system", "llm_provider", "model_name",
                "scenario_name", "session_id", "rule_compliance_score",
                "violated_rules_count",
            ]
        ].sort_values("rule_compliance_score"),
        width='stretch',
    )


# ---- Raw record viewer ---------------------------------------------------

st.subheader("Raw record viewer")
if "source_file" in f.columns and len(f["source_file"].unique()) > 0:
    file_choice = st.selectbox(
        "Source JSONL", sorted(f["source_file"].unique().tolist())
    )
    rows: list[dict] = []
    with open(file_choice, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    if rows:
        idx = st.slider("Record", 0, len(rows) - 1, 0)
        st.json(rows[idx])

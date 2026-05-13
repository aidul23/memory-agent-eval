"""Aggregate raw JSONL logs into a tidy DataFrame + condition-level CSVs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


def load_runs(
    results_dir: str | Path,
    *,
    experiment_id: str | None = None,
) -> pd.DataFrame:
    """Read every *.jsonl file under ``results_dir`` and return a long-form DataFrame.

    Each row is one (experiment, run, task, session) interaction.

    If ``experiment_id`` is provided, only rows belonging to that experiment
    are kept. This is the recommended way to scope analysis to a single run
    without manually managing the ``raw_logs/`` directory.
    """
    p = Path(results_dir)
    rows: list[dict[str, Any]] = []
    for fp in sorted(p.glob("**/*.jsonl")):
        with fp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if experiment_id and obj.get("experiment_id") != experiment_id:
                    continue
                rows.append(_flatten(obj, source=str(fp)))
    if not rows:
        scope = f" matching experiment_id={experiment_id!r}" if experiment_id else ""
        logger.warning("No JSONL records found under %s%s", results_dir, scope)
    return pd.DataFrame(rows)


def _flatten(rec: dict[str, Any], *, source: str) -> dict[str, Any]:
    ev = rec.get("evaluation_result") or {}
    fb = rec.get("feedback") or {}
    tu = rec.get("token_usage") or {}
    muq = ev.get("memory_usage_quality") or fb.get("memory_usage_quality") or {}
    return {
        "source_file": source,
        "experiment_id": rec.get("experiment_id"),
        "run_id": rec.get("run_id"),
        "run_index": rec.get("run_index"),
        "task_id": rec.get("task_id"),
        "scenario_name": rec.get("scenario_name"),
        "session_id": rec.get("session_id"),
        "memory_system": rec.get("memory_system"),
        "llm_provider": rec.get("llm_provider"),
        "model_name": rec.get("model_name"),
        "temperature": rec.get("temperature"),
        "task_success": bool(ev.get("task_success", False)),
        "rule_compliance_score": ev.get("rule_compliance_score"),
        "progress_score": ev.get("progress_score"),
        "violated_rules_count": len(ev.get("violated_rules", []) or []),
        "correct_subtasks": ev.get("correct_subtasks"),
        "total_subtasks": ev.get("total_subtasks"),
        "retrieval_relevance": muq.get("retrieval_relevance", 0.0),
        "retrieval_correctness": muq.get("retrieval_correctness", 0.0),
        "usage_quality": muq.get("usage_quality", 0.0),
        "input_tokens": tu.get("input_tokens", 0),
        "output_tokens": tu.get("output_tokens", 0),
        "estimated_cost_usd": tu.get("estimated_cost_usd", 0.0),
        "latency_s": rec.get("latency_s", 0.0),
        "timestamp": rec.get("timestamp"),
    }


def aggregate(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Produce condition-level CSV summaries + the long-form frame.

    If ``experiment_id`` is provided, restricts the aggregation to rows
    from that experiment only. This is the way to keep separate
    experimental runs from polluting each other's statistics.

    Returns paths to the CSVs produced, plus a small textual summary.
    """
    out = Path(output_dir)
    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(results_dir, experiment_id=experiment_id)
    frame_path = metrics_dir / "interactions.csv"
    df.to_csv(frame_path, index=False)

    summary: dict[str, Any] = {"interactions": len(df), "frame_path": str(frame_path)}
    if df.empty:
        return summary

    # Condition-level summary: memory_system x llm_provider x model_name.
    by_condition = (
        df.groupby(["memory_system", "llm_provider", "model_name"], dropna=False)
        .agg(
            n=("task_success", "size"),
            success_rate=("task_success", "mean"),
            mean_compliance=("rule_compliance_score", "mean"),
            std_compliance=("rule_compliance_score", "std"),
            mean_progress=("progress_score", "mean"),
            mean_relevance=("retrieval_relevance", "mean"),
            mean_correctness=("retrieval_correctness", "mean"),
            mean_usage=("usage_quality", "mean"),
            mean_latency=("latency_s", "mean"),
            mean_input_tokens=("input_tokens", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
            mean_cost_usd=("estimated_cost_usd", "mean"),
        )
        .reset_index()
    )
    by_condition_path = metrics_dir / "by_condition.csv"
    by_condition.to_csv(by_condition_path, index=False)
    summary["by_condition_path"] = str(by_condition_path)

    # Iteration depth: scenario session_id is the iteration.
    by_iteration = (
        df.groupby(["memory_system", "session_id"], dropna=False)
        .agg(
            n=("task_success", "size"),
            success_rate=("task_success", "mean"),
            mean_compliance=("rule_compliance_score", "mean"),
        )
        .reset_index()
        .sort_values(["memory_system", "session_id"])
    )
    by_iteration_path = metrics_dir / "by_iteration.csv"
    by_iteration.to_csv(by_iteration_path, index=False)
    summary["by_iteration_path"] = str(by_iteration_path)

    # Improvement slope per memory system: regress compliance on session_id.
    slopes = []
    for mem, sub in df.groupby("memory_system"):
        if sub["session_id"].nunique() < 2:
            slope = float("nan")
        else:
            x = sub["session_id"].astype(float).to_numpy()
            y = sub["rule_compliance_score"].astype(float).to_numpy()
            try:
                # Polyfit returns highest-degree-first.
                slope = float(_polyfit_slope(x, y))
            except Exception:
                slope = float("nan")
        slopes.append({"memory_system": mem, "improvement_slope": slope})
    slope_df = pd.DataFrame(slopes)
    slope_path = metrics_dir / "improvement_slope.csv"
    slope_df.to_csv(slope_path, index=False)
    summary["improvement_slope_path"] = str(slope_path)

    # Stateless baseline lift: how much each memory beats stateless on compliance.
    baseline = (
        df[df["memory_system"] == "stateless"]["rule_compliance_score"].mean()
        if (df["memory_system"] == "stateless").any()
        else None
    )
    if baseline is not None:
        lift = (
            df.groupby("memory_system")["rule_compliance_score"]
            .mean()
            .reset_index(name="mean_compliance")
        )
        lift["baseline_lift"] = lift["mean_compliance"] - baseline
        lift_path = metrics_dir / "stateless_lift.csv"
        lift.to_csv(lift_path, index=False)
        summary["stateless_lift_path"] = str(lift_path)
        summary["stateless_baseline_compliance"] = baseline

    return summary


def _polyfit_slope(x, y) -> float:
    import numpy as np

    coeffs = np.polyfit(x, y, deg=1)
    return float(coeffs[0])

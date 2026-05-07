"""Plot the canonical research-question-aligned figures.

Each figure answers a specific RQ:
- success_by_memory.png  -> RQ1
- compliance_by_iteration.png -> RQ2
- memory_utility.png -> RQ4
- model_comparison.png -> RQ3
- stateless_vs_augmented.png -> RQ2
- improvement_slope.png -> RQ2

Plots are intentionally simple matplotlib figures with no global style hacks
so they reproduce identically in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


def render_all(frame_csv: str | Path, output_dir: str | Path) -> list[str]:
    """Render every canonical figure and return the file paths produced."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    df = pd.read_csv(frame_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []

    if df.empty:
        logger.warning("No interactions to plot.")
        return produced

    produced.append(_bar_success_by_memory(df, out_dir, plt))
    produced.append(_line_compliance_by_iteration(df, out_dir, plt))
    produced.append(_bar_memory_utility(df, out_dir, plt))
    produced.append(_bar_model_comparison(df, out_dir, plt))
    produced.append(_bar_stateless_vs_augmented(df, out_dir, plt))
    produced.append(_bar_improvement_slope(df, out_dir, plt))
    return [p for p in produced if p]


def _bar_success_by_memory(df: pd.DataFrame, out_dir: Path, plt) -> str:
    grouped = df.groupby("memory_system")["task_success"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped.plot(kind="bar", ax=ax, color="#3f7eba")
    ax.set_ylabel("Task success rate")
    ax.set_xlabel("Memory system")
    ax.set_title("Task success rate by memory system (RQ1)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    p = out_dir / "success_by_memory.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def _line_compliance_by_iteration(df: pd.DataFrame, out_dir: Path, plt) -> str:
    pivot = (
        df.groupby(["session_id", "memory_system"])["rule_compliance_score"]
        .mean()
        .unstack("memory_system")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(ax=ax, marker="o")
    ax.set_xlabel("Session (iteration depth)")
    ax.set_ylabel("Mean rule compliance")
    ax.set_title("Rule compliance over iterations by memory system (RQ2)")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Memory")
    fig.tight_layout()
    p = out_dir / "compliance_by_iteration.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def _bar_memory_utility(df: pd.DataFrame, out_dir: Path, plt) -> str:
    cols = ["retrieval_relevance", "retrieval_correctness", "usage_quality"]
    sub = df.groupby("memory_system")[cols].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    sub.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score (0-1)")
    ax.set_xlabel("Memory system")
    ax.set_title("Memory utility components by memory system (RQ4)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = out_dir / "memory_utility.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def _bar_model_comparison(df: pd.DataFrame, out_dir: Path, plt) -> str:
    if df["llm_provider"].nunique() < 2 and df["model_name"].nunique() < 2:
        # Still render, but with a friendly note.
        pass
    sub = df.groupby(["llm_provider", "model_name"])["rule_compliance_score"].mean().reset_index()
    sub["label"] = sub["llm_provider"].astype(str) + "/" + sub["model_name"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sub["label"], sub["rule_compliance_score"], color="#7e57c2")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["label"], rotation=20, ha="right")
    ax.set_ylabel("Mean rule compliance")
    ax.set_title("Model comparison (RQ3)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = out_dir / "model_comparison.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def _bar_stateless_vs_augmented(df: pd.DataFrame, out_dir: Path, plt) -> str:
    df = df.copy()
    df["augmented"] = df["memory_system"].apply(lambda m: "stateless" if m == "stateless" else "memory-augmented")
    sub = df.groupby("augmented")["rule_compliance_score"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    sub.plot(kind="bar", ax=ax, color=["#999", "#2e7d32"])
    ax.set_ylabel("Mean rule compliance")
    ax.set_title("Stateless vs memory-augmented (RQ2)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = out_dir / "stateless_vs_augmented.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


def _bar_improvement_slope(df: pd.DataFrame, out_dir: Path, plt) -> str:
    import numpy as np

    rows = []
    for mem, sub in df.groupby("memory_system"):
        if sub["session_id"].nunique() < 2:
            slope = float("nan")
        else:
            x = sub["session_id"].astype(float).to_numpy()
            y = sub["rule_compliance_score"].astype(float).to_numpy()
            try:
                slope = float(np.polyfit(x, y, 1)[0])
            except Exception:
                slope = float("nan")
        rows.append((mem, slope))
    series = pd.Series({m: s for m, s in rows}).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    series.plot(kind="bar", ax=ax, color="#ef6c00")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Slope of compliance over sessions")
    ax.set_title("Improvement slope per memory system (RQ2)")
    fig.tight_layout()
    p = out_dir / "improvement_slope.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return str(p)


__all__: Iterable[str] = ("render_all",)

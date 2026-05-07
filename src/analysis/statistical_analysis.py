"""Statistical comparisons between memory systems and LLM providers.

We use scipy / statsmodels for:
- One-way ANOVA across memory_system on rule_compliance_score.
- One-way ANOVA across llm_provider on rule_compliance_score.
- Two-way ANOVA on memory_system x llm_provider (interaction effect).
- Pairwise Tukey HSD post-hoc.
- Cohen's d effect size for each memory system vs the stateless baseline.

All tests are guarded against degenerate inputs (single group, NaNs) and
return ``None`` rather than raising, so downstream code (CLI, dashboard) can
report "n/a" for under-powered conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


def _safe_anova(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any] | None:
    from scipy import stats

    groups = [g[value_col].dropna().to_numpy() for _, g in df.groupby(group_col)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return None
    if all(np.allclose(g.std(), 0.0) for g in groups):
        return {"f_stat": 0.0, "p_value": 1.0, "note": "All groups have zero variance."}
    f, p = stats.f_oneway(*groups)
    return {"f_stat": float(f), "p_value": float(p), "n_groups": len(groups)}


def _safe_two_way_anova(df: pd.DataFrame, value_col: str) -> dict[str, Any] | None:
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        return None

    sub = df[[value_col, "memory_system", "llm_provider"]].dropna()
    if sub["memory_system"].nunique() < 2 or sub["llm_provider"].nunique() < 2:
        return None
    try:
        model = ols(
            f"{value_col} ~ C(memory_system) + C(llm_provider) + C(memory_system):C(llm_provider)",
            data=sub,
        ).fit()
        table = anova_lm(model, typ=2)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return {
        "table": table.reset_index().rename(columns={"index": "term"}).to_dict(orient="records"),
    }


def _safe_tukey(df: pd.DataFrame, group_col: str, value_col: str) -> list[dict[str, Any]] | None:
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError:
        return None
    sub = df[[value_col, group_col]].dropna()
    if sub[group_col].nunique() < 2:
        return None
    res = pairwise_tukeyhsd(endog=sub[value_col], groups=sub[group_col], alpha=0.05)
    rows: list[dict[str, Any]] = []
    for line in res._results_table.data[1:]:
        rows.append({
            "group1": line[0],
            "group2": line[1],
            "mean_diff": float(line[2]),
            "p_adj": float(line[3]),
            "lower": float(line[4]),
            "upper": float(line[5]),
            "reject": bool(line[6]),
        })
    return rows


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                     (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def run_basic_stats(frame_csv: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run the canonical battery of tests on the aggregated DataFrame."""
    df = pd.read_csv(frame_csv)
    out: dict[str, Any] = {"n_rows": len(df)}
    if df.empty:
        return out

    out["anova_memory_system_on_compliance"] = _safe_anova(df, "memory_system", "rule_compliance_score")
    out["anova_llm_provider_on_compliance"] = _safe_anova(df, "llm_provider", "rule_compliance_score")
    out["two_way_memory_x_llm_on_compliance"] = _safe_two_way_anova(df, "rule_compliance_score")
    out["tukey_memory_system_on_compliance"] = _safe_tukey(df, "memory_system", "rule_compliance_score")
    out["tukey_llm_provider_on_compliance"] = _safe_tukey(df, "llm_provider", "rule_compliance_score")

    # Cohen's d vs stateless baseline.
    baseline = df[df["memory_system"] == "stateless"]["rule_compliance_score"].dropna().to_numpy()
    effects = []
    for mem, sub in df.groupby("memory_system"):
        if mem == "stateless":
            continue
        treat = sub["rule_compliance_score"].dropna().to_numpy()
        effects.append({
            "memory_system": mem,
            "n_treatment": int(len(treat)),
            "n_baseline": int(len(baseline)),
            "cohens_d_vs_stateless": _cohens_d(treat, baseline) if len(baseline) else float("nan"),
        })
    out["effect_sizes_vs_stateless"] = effects

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics" / "stats.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics" / "stats.json").write_text(json.dumps(out, indent=2, default=str))
    out["stats_path"] = str(out_dir / "metrics" / "stats.json")
    return out

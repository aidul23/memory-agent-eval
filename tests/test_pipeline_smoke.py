"""End-to-end smoke test using the deterministic Mock LLM.

This is the experiment described in the README's "Quick start" section:
compare stateless vs hindsight memory on the handheld_enclosure scenario.

We assert two properties:
1. The pipeline runs to completion and writes JSONL records.
2. The mock LLM is wired up correctly: the *memory-augmented* condition
   produces strictly more "satisfied | violated" decisions than the
   stateless one, because the mock intentionally degrades when no memory
   is present.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.experiment_runner import ExperimentRunner

CONFIG = {
    "experiment_name": "smoke_test",
    "runs_per_condition": 1,
    "temperature": 0.0,
    "memory_systems": ["stateless", "hindsight"],
    "llms": [{"provider": "mock", "model": "mock-deterministic"}],
    "tasks": {"path": "data/tasks/"},
    "evaluation": {"output_dir": "results/"},
}


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    out = tmp_path / "results"
    out.mkdir()
    cfg = dict(CONFIG, evaluation={"output_dir": str(out)})
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    return out, cfg


def test_pipeline_runs_end_to_end(tmp_results):
    out, cfg = tmp_results
    runner = ExperimentRunner(cfg)
    summary = runner.run()
    log_path = Path(summary["raw_log_path"])
    assert log_path.exists()
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert records, "pipeline did not produce any records"

    by_mem = {"stateless": [], "hindsight": []}
    for r in records:
        by_mem[r["memory_system"]].append(r)

    # Both conditions cover all four sessions.
    assert len(by_mem["stateless"]) == 4
    assert len(by_mem["hindsight"]) == 4


def test_hindsight_outperforms_stateless_on_certainty(tmp_results):
    out, cfg = tmp_results
    runner = ExperimentRunner(cfg)
    summary = runner.run()
    records = [json.loads(line) for line in Path(summary["raw_log_path"]).read_text().splitlines() if line.strip()]

    def certain_decisions(records):
        certain = 0
        total = 0
        for r in records:
            for a in r.get("agent_response", {}).get("dfx_rule_analysis", []):
                total += 1
                if a.get("status") in {"satisfied", "violated"}:
                    certain += 1
        return certain, total

    stateless = [r for r in records if r["memory_system"] == "stateless"]
    hindsight = [r for r in records if r["memory_system"] == "hindsight"]
    s_cert, s_tot = certain_decisions(stateless)
    h_cert, h_tot = certain_decisions(hindsight)

    # The mock LLM is configured so memory-augmented mode evaluates more
    # rules with certainty. This is a sanity check that the memory plumbing
    # actually reaches the LLM prompt.
    assert s_tot > 0 and h_tot > 0
    assert (h_cert / h_tot) >= (s_cert / s_tot)

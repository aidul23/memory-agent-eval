"""Test the end-to-end evaluator on the canonical sample tasks."""

from pathlib import Path

from src.evaluation.evaluator import Evaluator
from src.tasks.task_loader import TaskLoader

TASKS_DIR = Path("data/tasks")


def _load(task_filename: str):
    return TaskLoader(TASKS_DIR).load_task(TASKS_DIR / task_filename)


def test_session_1_perfect_agent_reports_all_violations():
    task = _load("enclosure_dfm_session_1.yaml")
    evaluator = Evaluator()
    perfect_response = {
        "summary": "All five rules violated.",
        "decision": "reject",
        "dfx_rule_analysis": [
            {"rule_id": rid, "status": "violated", "explanation": "ground truth"}
            for rid in ["DFM-001", "DFM-002", "DFM-003", "DFM-004", "DFM-005"]
        ],
        "used_memory": [],
        "final_recommendation": "fix everything",
        "confidence": 0.95,
    }
    res = evaluator.evaluate(task=task, agent_response=perfect_response, retrieved_memory=[])
    assert res.task_success is False, "design itself fails - success refers to compliance"
    assert res.rule_compliance_score == 0.0
    assert set(res.violated_rules) == {"DFM-001", "DFM-002", "DFM-003", "DFM-004", "DFM-005"}
    assert res.correct_subtasks == 5  # agent labelled all five correctly


def test_session_3_clean_design_passes():
    task = _load("enclosure_dfm_session_3.yaml")
    evaluator = Evaluator()
    response = {
        "summary": "All rules satisfied.",
        "decision": "approve",
        "dfx_rule_analysis": [
            {"rule_id": rid, "status": "satisfied", "explanation": "ok"}
            for rid in ["DFM-001", "DFM-002", "DFM-003", "DFM-004", "DFM-005"]
        ],
        "used_memory": [],
        "final_recommendation": "proceed",
        "confidence": 0.99,
    }
    res = evaluator.evaluate(task=task, agent_response=response, retrieved_memory=[])
    assert res.task_success is True
    assert res.rule_compliance_score == 1.0
    assert res.violated_rules == []

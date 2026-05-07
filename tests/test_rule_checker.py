"""Tests for the deterministic rule checker."""

from src.evaluation.rule_checker import RuleChecker
from src.tasks.task_loader import load_rule_pack


def test_dfm_pack_loads():
    pack = load_rule_pack("data/rules/dfm_rules.yaml")
    assert pack.rule_pack == "dfm"
    assert {r.id for r in pack.rules} == {
        "DFM-001", "DFM-002", "DFM-003", "DFM-004", "DFM-005",
    }


def test_session_1_design_violates_all_rules():
    pack = load_rule_pack("data/rules/dfm_rules.yaml")
    design = {
        "wall_thickness_mm": 1.1,
        "draft_angle_deg": 0.5,
        "has_undercut": True,
        "internal_radius_mm": 0.2,
        "boss_outer_diameter_mm": 4.5,
    }
    results = RuleChecker(pack).check_all(design)
    statuses = {r.rule_id: r.status for r in results}
    assert all(s == "violated" for s in statuses.values()), statuses


def test_session_3_design_passes_everything():
    pack = load_rule_pack("data/rules/dfm_rules.yaml")
    design = {
        "wall_thickness_mm": 1.6,
        "draft_angle_deg": 1.2,
        "has_undercut": False,
        "internal_radius_mm": 0.6,
        "boss_outer_diameter_mm": 3.6,
    }
    results = RuleChecker(pack).check_all(design)
    assert all(r.status == "satisfied" for r in results), [r.to_dict() for r in results]


def test_session_2_design_only_violates_internal_radius():
    pack = load_rule_pack("data/rules/dfm_rules.yaml")
    design = {
        "wall_thickness_mm": 1.6,
        "draft_angle_deg": 1.2,
        "has_undercut": False,
        "internal_radius_mm": 0.4,
        "boss_outer_diameter_mm": 3.6,
    }
    results = RuleChecker(pack).check_all(design)
    violated = [r.rule_id for r in results if r.status == "violated"]
    assert violated == ["DFM-004"]

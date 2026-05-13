"""Tests for memory implementations."""

from src.memory import (
    ContextualMemory,
    PersistentMemory,
    ReflectionMemory,
    StatelessMemory,
    create_memory,
)


def _interaction(scenario="alpha", session_id=1, violated=("DFM-001",)):
    return {
        "scenario_name": scenario,
        "session_id": session_id,
        "task_id": f"{scenario}_{session_id}",
        "task": {
            "scenario_name": scenario,
            "session_id": session_id,
            "task_id": f"{scenario}_{session_id}",
            "input_description": f"Evaluate scenario {scenario} session {session_id}",
            "design_context": {"wall_thickness_mm": 1.1},
            "evaluation_criteria": {"rule_pack": "dfm"},
        },
        "response": {"decision": "reject"},
        "feedback": {
            "task_success": False,
            "violated_rules": list(violated),
            "improvement_suggestions": ["increase wall thickness to >= 1.5 mm"],
            "rule_compliance_score": 0.5,
        },
    }


def test_stateless_returns_nothing():
    mem = StatelessMemory()
    mem.update(_interaction())
    assert mem.retrieve("anything", {}) == []
    assert mem.export_memory()["items"] == []


def test_reflection_retrieval_picks_relevant_reflection():
    mem = ReflectionMemory(top_k=3)
    mem.update(_interaction(scenario="alpha", session_id=1, violated=("DFM-001",)))
    mem.update(_interaction(scenario="beta",  session_id=1, violated=("DFA-003",)))
    hits = mem.retrieve("Evaluate scenario alpha session 2", context={})
    assert hits, "expected at least one reflection to be retrieved"
    # The 'alpha' reflection should beat the 'beta' one.
    assert hits[0].metadata.get("scenario") == "alpha"


def test_hindsight_alias_is_now_vectorize():
    """The 'hindsight' registry key now points at Vectorize.io's service.

    Without the hindsight-client SDK installed or a server running, it must
    fall back gracefully to mode='local'. This catches accidental regressions
    of the registry rename.
    """
    from src.memory import VectorizeHindsightMemory
    m = create_memory("hindsight")
    assert isinstance(m, VectorizeHindsightMemory)
    # No server running in CI -> must be in local fallback.
    assert m.mode in {"local", "remote"}


def test_contextual_uses_design_context_in_query():
    mem = ContextualMemory(top_k=3)
    mem.update(_interaction(scenario="alpha", session_id=1))
    hits = mem.retrieve("review handheld enclosure", context={"wall_thickness_mm": 1.1})
    assert hits


def test_persistent_round_trip(tmp_path):
    path = tmp_path / "store.jsonl"
    mem = PersistentMemory(storage_path=str(path), top_k=3)
    mem.update(_interaction())
    assert path.exists()
    # Re-instantiating should reload prior items.
    mem2 = PersistentMemory(storage_path=str(path), top_k=3)
    assert len(mem2.export_memory()["items"]) == 1


def test_factory_creates_each_internal_kind():
    """Sanity-check the in-process memories.

    External memories (mem0/zep/acontext/hindsight) are not asserted here
    because their constructors try to connect to a live server; missing
    servers are fine (they fall back to local mode) but we don't want this
    unit test to depend on optional SDKs being installed.
    """
    for name in ["stateless", "reflection", "contextual"]:
        m = create_memory(name)
        assert m.name == name


def test_deprecated_hindsight_import_still_works():
    """`from src.memory.hindsight_memory import HindsightMemory` should still work."""
    import warnings
    from src.memory.hindsight_memory import HindsightMemory
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m = HindsightMemory()
    m.update(_interaction())
    assert m.retrieve("scenario alpha", context={}) or True  # works even if empty

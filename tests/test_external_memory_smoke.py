"""Live smoke tests for the four external memory backends.

Each test is auto-skipped when:
- the corresponding SDK package is not installed, or
- the corresponding server is not reachable on its expected URL.

If the server is up, the test does a minimal:
  1. instantiate memory
  2. assert ``mode == 'remote'``
  3. call ``update()`` once
  4. call ``retrieve()`` and assert at least zero items came back (no crash)
  5. call ``reset()`` (uses the configured reset_policy)

This catches API-shape regressions without requiring docker-compose at CI.
Run with the relevant servers up: ``pytest tests/test_external_memory_smoke.py``.
"""

from __future__ import annotations

import importlib
import os
import socket
from urllib.parse import urlparse

import pytest

from src.memory import (
    AContextMemory,
    Mem0Memory,
    VectorizeHindsightMemory,
    ZepMemory,
)


def _sdk_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _server_up(url: str | None, *, timeout: float = 1.0) -> bool:
    """Return True iff a TCP connection to the URL's host:port succeeds."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


INTERACTION = {
    "scenario_name": "smoke",
    "session_id": 1,
    "task_id": "smoke_1",
    "run_id": "smoke_test_run",
    "task": {
        "scenario_name": "smoke",
        "session_id": 1,
        "task_id": "smoke_1",
        "input_description": "Smoke test memory write.",
        "design_context": {"wall_thickness_mm": 1.1},
        "evaluation_criteria": {"rule_pack": "dfm"},
    },
    "response": {"decision": "reject"},
    "feedback": {
        "task_success": False,
        "violated_rules": ["DFM-001"],
        "improvement_suggestions": ["Increase wall thickness to 1.6mm."],
        "rule_compliance_score": 0.5,
    },
}


# -------------------- Mem0 --------------------

@pytest.mark.skipif(not _sdk_available("mem0"), reason="mem0ai SDK not installed")
@pytest.mark.skipif(
    not _server_up(os.getenv("MEM0_QDRANT_URL", "http://localhost:6333")),
    reason="Qdrant not running on MEM0_QDRANT_URL",
)
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set (Mem0 needs an LLM)")
def test_mem0_round_trip():
    with Mem0Memory(user_id="dfx_smoke", top_k=3, reset_policy="clear_remote") as m:
        assert m.mode == "remote", "Mem0 connection did not establish"
        m.update(INTERACTION)
        hits = m.retrieve("Smoke test", context={"run_id": INTERACTION["run_id"]})
        assert isinstance(hits, list)
        m.reset()


# -------------------- Zep --------------------

@pytest.mark.skipif(not _sdk_available("zep_cloud"), reason="zep-cloud SDK not installed")
@pytest.mark.skipif(
    not _server_up(os.getenv("ZEP_API_URL", "http://localhost:8000")),
    reason="Zep server not running on ZEP_API_URL",
)
def test_zep_round_trip():
    with ZepMemory(session_id_prefix="dfx-smoke", top_k=3, reset_policy="clear_remote") as m:
        assert m.mode == "remote", "Zep connection did not establish"
        m.update(INTERACTION)
        hits = m.retrieve("Smoke test", context={
            "run_id": INTERACTION["run_id"],
            "scenario_name": INTERACTION["scenario_name"],
        })
        assert isinstance(hits, list)
        m.reset()


# -------------------- AContext --------------------

@pytest.mark.skipif(not _sdk_available("acontext"), reason="acontext SDK not installed")
@pytest.mark.skipif(
    not _server_up(os.getenv("ACONTEXT_API_URL", "http://localhost:8029/api/v1")),
    reason="AContext server not running on ACONTEXT_API_URL",
)
@pytest.mark.skipif(not os.getenv("ACONTEXT_API_KEY"), reason="ACONTEXT_API_KEY not set")
def test_acontext_round_trip():
    with AContextMemory(workspace="dfx_smoke", top_k=3, reset_policy="clear_remote") as m:
        assert m.mode == "remote", "AContext connection did not establish"
        m.update(INTERACTION)
        hits = m.retrieve("Smoke test", context={
            "run_id": INTERACTION["run_id"],
            "scenario_name": INTERACTION["scenario_name"],
        })
        assert isinstance(hits, list)
        m.reset()


# -------------------- Vectorize Hindsight --------------------

@pytest.mark.skipif(not _sdk_available("hindsight_client"), reason="hindsight-client SDK not installed")
@pytest.mark.skipif(
    not _server_up(os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")),
    reason="Vectorize Hindsight server not running on HINDSIGHT_API_URL",
)
def test_vectorize_hindsight_round_trip():
    with VectorizeHindsightMemory(bank_prefix="dfx-smoke", top_k=3, reset_policy="clear_remote") as m:
        assert m.mode == "remote", "Hindsight connection did not establish"
        m.update(INTERACTION)
        hits = m.retrieve("Smoke test", context={
            "run_id": INTERACTION["run_id"],
            "scenario_name": INTERACTION["scenario_name"],
        })
        assert isinstance(hits, list)
        m.reset()


# -------------------- Always-on contract tests --------------------

def test_all_external_wrappers_fall_back_gracefully_when_offline():
    """Even with no servers running and no SDKs installed, the wrappers
    must instantiate without raising and report mode='local' (or 'remote'
    when the SDK's constructor is lazy enough to not detect the dead URL).

    We exercise the context manager so transport resources are always
    released, even when the underlying SDK leaks aiohttp sessions on
    partial failure.
    """
    os.environ.setdefault("HINDSIGHT_API_URL", "http://localhost:1")
    for cls in [Mem0Memory, ZepMemory, AContextMemory, VectorizeHindsightMemory]:
        with cls(top_k=2, api_key="invalid", base_url="http://localhost:1") as m:  # type: ignore[arg-type]
            assert m.mode in {"local", "remote"}, m
            m.update(INTERACTION)
            hits = m.retrieve("Smoke test", context={"run_id": "x", "scenario_name": "smoke"})
            assert isinstance(hits, list)


def test_reset_policy_validation():
    """Bad reset_policy values must fail loudly."""
    with pytest.raises(ValueError):
        Mem0Memory(reset_policy="invalid_policy")  # type: ignore[arg-type]


def test_reset_policy_keep_remote_does_not_call_remote_clear(monkeypatch):
    """With keep_remote, _remote_clear must NOT be invoked."""
    m = Mem0Memory(reset_policy="keep_remote", api_key="x", base_url="http://localhost:1")  # type: ignore[arg-type]
    called = {"yes": False}

    def fake_clear():
        called["yes"] = True

    m._remote_clear = fake_clear  # type: ignore[method-assign]
    # Force remote mode for the test even though _connect failed.
    m._mode = "remote"
    m.reset()
    assert called["yes"] is False

"""Experiment orchestrator.

Cross-product:  memory_systems x llms x scenarios x runs_per_condition.

For every (memory, llm) condition we replay each scenario from session_1
upwards, calling the agent, evaluating the response, generating feedback,
and updating the memory. Every interaction is logged as one JSONL record.

Reset semantics
---------------
- Memory is RESET between repeated runs of the same condition (so each
  ``run_index`` is an independent learning trajectory).
- Memory is also RESET between different *scenarios* by default; configure
  ``cross_scenario_memory: true`` in experiment.yaml to test transfer.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .agents.memory_agent import MemoryAgent
from .evaluation.evaluator import Evaluator
from .llms import create_llm
from .llms.base_llm import BaseLLM
from .memory import create_memory
from .memory.base_memory import BaseMemory
from .tasks.dfx_task import DFxTask
from .tasks.scenario_manager import ScenarioManager
from .tasks.task_loader import TaskLoader
from .utils import JsonlWriter, get_logger, load_config, merge_configs

logger = get_logger(__name__)


class ExperimentRunner:
    """Top-level orchestrator. Reads a config, runs every condition, logs everything."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.experiment_name = config.get("experiment_name", "unnamed_experiment")
        self.experiment_id = f"{self.experiment_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
        self.runs_per_condition = int(config.get("runs_per_condition", 1))
        self.temperature = float(config.get("temperature", 0.0))
        self.seed = int(config.get("seed", 0))
        self.cross_scenario_memory = bool(config.get("cross_scenario_memory", False))

        ev_cfg = config.get("evaluation", {}) or {}
        self.output_dir = Path(ev_cfg.get("output_dir", "results/"))
        self.raw_log_path = self.output_dir / "raw_logs" / f"{self.experiment_id}.jsonl"
        self.summary_path = self.output_dir / "raw_logs" / f"{self.experiment_id}.summary.json"
        self.writer = JsonlWriter(self.raw_log_path)

        # Memory- and model-specific defaults loaded from sibling configs.
        self.memory_defaults = self._load_optional("configs/memory_systems.yaml")
        self.model_defaults = self._load_optional("configs/models.yaml")

    # ---- Public API --------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run every condition. Returns a summary dict."""
        tasks_cfg = self.config.get("tasks", {})
        loader = TaskLoader(tasks_cfg.get("path", "data/tasks/"))
        all_tasks = loader.load_all()
        tasks = TaskLoader.filter_tasks(
            all_tasks,
            include_scenarios=tasks_cfg.get("include_scenarios"),
            exclude_scenarios=tasks_cfg.get("exclude_scenarios"),
        )
        scenarios = ScenarioManager(tasks)
        if not len(scenarios):
            raise RuntimeError(f"No tasks discovered under {tasks_cfg.get('path')!r}")

        memory_names = list(self.config.get("memory_systems") or ["stateless"])
        llm_specs = list(self.config.get("llms") or [{"provider": "mock", "model": "mock-deterministic"}])

        evaluator = Evaluator()
        n_interactions = 0
        condition_count = 0
        t_start = time.perf_counter()

        for memory_name in memory_names:
            for llm_spec in llm_specs:
                for run_index in range(self.runs_per_condition):
                    condition_count += 1
                    run_id = f"{memory_name}__{llm_spec.get('provider')}__{llm_spec.get('model')}__r{run_index}"
                    memory = self._build_memory(memory_name)
                    llm = self._build_llm(llm_spec)
                    agent = MemoryAgent(llm=llm, memory=memory, temperature=self.temperature)

                    logger.info(
                        "Running condition %s (memory=%s, provider=%s, model=%s, run %d/%d)",
                        run_id, memory_name, llm_spec.get("provider"), llm_spec.get("model"),
                        run_index + 1, self.runs_per_condition,
                    )

                    try:
                        for scenario in scenarios:
                            if not self.cross_scenario_memory:
                                memory.reset()
                            prior_feedback: dict[str, Any] | None = None
                            for task in scenario.sessions:
                                n_interactions += 1
                                self._run_one(
                                    run_id=run_id,
                                    run_index=run_index,
                                    memory_name=memory_name,
                                    llm_spec=llm_spec,
                                    memory=memory,
                                    agent=agent,
                                    evaluator=evaluator,
                                    task=task,
                                    prior_feedback=prior_feedback,
                                )
                                # Reload the last log entry's feedback for the next session.
                                # (We do this via the agent path inside _run_one return value.)
                                prior_feedback = self._last_feedback
                    finally:
                        # Release transport resources held by external memory
                        # SDKs (httpx/aiohttp). Safe to call on any BaseMemory:
                        # ExternalMemoryBase implements close(); in-process
                        # memories ignore it via the getattr fallback below.
                        closer = getattr(memory, "close", None)
                        if callable(closer):
                            try:
                                closer()
                            except Exception:  # noqa: BLE001
                                logger.debug("memory.close() raised; ignoring.")

        elapsed = time.perf_counter() - t_start
        summary = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "conditions": condition_count,
            "interactions": n_interactions,
            "elapsed_s": round(elapsed, 3),
            "raw_log_path": str(self.raw_log_path),
        }
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(__import__("json").dumps(summary, indent=2))
        logger.info("Experiment %s done in %.1fs - %d interactions logged to %s",
                    self.experiment_id, elapsed, n_interactions, self.raw_log_path)
        return summary

    # ---- One interaction --------------------------------------------

    _last_feedback: dict[str, Any] | None = None

    def _run_one(
        self,
        *,
        run_id: str,
        run_index: int,
        memory_name: str,
        llm_spec: dict[str, Any],
        memory: BaseMemory,
        agent: MemoryAgent,
        evaluator: Evaluator,
        task: DFxTask,
        prior_feedback: dict[str, Any] | None,
    ) -> None:
        agent_result = agent.run(task, prior_feedback=prior_feedback)
        eval_result = evaluator.evaluate(
            task=task,
            agent_response=agent_result.response,
            retrieved_memory=agent_result.retrieved_memory,
        )

        interaction = {
            "task": task.model_dump(),
            "response": agent_result.response,
            "feedback": eval_result.feedback,
            "scenario_name": task.scenario_name,
            "session_id": task.session_id,
            "task_id": task.task_id,
            # run_id lets external memories scope sessions/banks per-run, so a
            # condition's run #0 cannot read run #1's data.
            "run_id": run_id,
            "run_index": run_index,
            "experiment_id": self.experiment_id,
        }
        # Update memory AFTER evaluation so it sees canonical feedback.
        memory.update(interaction)

        llm_resp = agent_result.llm_response
        record = {
            "experiment_id": self.experiment_id,
            "run_id": run_id,
            "run_index": run_index,
            "task_id": task.task_id,
            "scenario_name": task.scenario_name,
            "session_id": task.session_id,
            "memory_system": memory_name,
            "llm_provider": llm_spec.get("provider"),
            "model_name": llm_spec.get("model"),
            "temperature": self.temperature,
            "seed": self.seed,
            "prompt": agent_result.prompt_messages,
            "retrieved_memory": agent_result.memory_as_dicts(),
            "agent_response": agent_result.response,
            "agent_raw_text": agent_result.raw_text,
            "evaluation_result": {
                "task_success": eval_result.task_success,
                "rule_compliance_score": eval_result.rule_compliance_score,
                "violated_rules": eval_result.violated_rules,
                "correct_subtasks": eval_result.correct_subtasks,
                "total_subtasks": eval_result.total_subtasks,
                "progress_score": eval_result.progress_score,
                "rule_results": [r.to_dict() for r in eval_result.rule_results],
                "memory_usage_quality": eval_result.memory_usage_quality,
            },
            "feedback": eval_result.feedback,
            "memory_update": {
                "memory_system": memory_name,
                "items_after_update": len(memory.export_memory().get("items", [])),
            },
            "latency_s": getattr(llm_resp, "latency_s", 0.0),
            "token_usage": {
                "input_tokens": getattr(llm_resp, "input_tokens", 0),
                "output_tokens": getattr(llm_resp, "output_tokens", 0),
                "estimated_cost_usd": getattr(llm_resp, "estimated_cost_usd", 0.0),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.writer.write(record)
        self._last_feedback = eval_result.feedback

    # ---- Construction helpers ----------------------------------------

    def _build_memory(self, name: str) -> BaseMemory:
        cfg = (self.memory_defaults or {}).get(name, {}) or {}
        return create_memory(name, **cfg)

    def _build_llm(self, spec: dict[str, Any]) -> BaseLLM:
        provider = spec["provider"]
        model = spec.get("model") or self._provider_default_model(provider)
        defaults = ((self.model_defaults or {}).get("providers") or {}).get(provider, {}) or {}
        kwargs = {
            "request_timeout": defaults.get("request_timeout", 60),
            "cost_per_1k_input_tokens": defaults.get("cost_per_1k_input_tokens"),
            "cost_per_1k_output_tokens": defaults.get("cost_per_1k_output_tokens"),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return create_llm(provider=provider, model=model, **kwargs)

    def _provider_default_model(self, provider: str) -> str:
        defaults = ((self.model_defaults or {}).get("providers") or {}).get(provider, {})
        return defaults.get("default_model", "mock-deterministic")

    @staticmethod
    def _load_optional(path: str | Path) -> dict[str, Any]:
        try:
            return load_config(path)
        except FileNotFoundError:
            return {}


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Public helper for the CLI. Layered on top of platform defaults."""
    base = {
        "experiment_name": "unnamed",
        "runs_per_condition": 1,
        "temperature": 0.0,
        "memory_systems": ["stateless"],
        "llms": [{"provider": "mock", "model": "mock-deterministic"}],
        "tasks": {"path": "data/tasks/"},
        "evaluation": {"output_dir": "results/"},
    }
    return merge_configs(base, load_config(path))

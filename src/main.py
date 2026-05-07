"""Command-line interface for the memory-agent evaluation platform.

Usage examples:

    python -m src.main run --config configs/experiment.yaml
    python -m src.main run-single --memory hindsight --llm mock \\
        --model mock-deterministic --task data/tasks/enclosure_dfm_session_1.yaml
    python -m src.main analyze --results results/raw_logs/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from .agents.memory_agent import MemoryAgent
from .evaluation.evaluator import Evaluator
from .experiment_runner import ExperimentRunner, load_experiment_config
from .llms import create_llm
from .memory import create_memory
from .tasks.task_loader import TaskLoader
from .utils import JsonlWriter, get_logger, load_config

logger = get_logger(__name__)


@click.group(help="Memory-agent evaluation platform CLI.")
def cli() -> None:
    load_dotenv()


@cli.command("run", help="Run a full experiment from a YAML config.")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              required=True, help="Path to experiment.yaml")
def cmd_run(config_path: str) -> None:
    cfg = load_experiment_config(config_path)
    runner = ExperimentRunner(cfg)
    summary = runner.run()
    click.echo(json.dumps(summary, indent=2))


@cli.command("run-single", help="Run a single (memory x llm x task) interaction. Useful for smoke-tests.")
@click.option("--memory", "memory_name", default="stateless", show_default=True)
@click.option("--llm", "llm_provider", default="mock", show_default=True)
@click.option("--model", "model_name", default="mock-deterministic", show_default=True)
@click.option("--task", "task_path", required=True, type=click.Path(exists=True))
@click.option("--temperature", default=0.0, type=float, show_default=True)
@click.option("--memory-config", "memory_config", default=None,
              help="Optional path to memory_systems.yaml; defaults to configs/memory_systems.yaml.")
@click.option("--output", "output_path", default=None,
              help="Optional path for a JSONL log of this single interaction.")
def cmd_run_single(
    memory_name: str,
    llm_provider: str,
    model_name: str,
    task_path: str,
    temperature: float,
    memory_config: str | None,
    output_path: str | None,
) -> None:
    load_dotenv()
    mem_defaults = _safe_load(memory_config or "configs/memory_systems.yaml")
    mem_kwargs = (mem_defaults.get(memory_name) if mem_defaults else {}) or {}

    memory = create_memory(memory_name, **mem_kwargs)
    llm = create_llm(provider=llm_provider, model=model_name)
    agent = MemoryAgent(llm=llm, memory=memory, temperature=temperature)
    evaluator = Evaluator()

    task = TaskLoader(Path(task_path).parent).load_task(task_path)
    result = agent.run(task)
    eval_res = evaluator.evaluate(
        task=task,
        agent_response=result.response,
        retrieved_memory=result.retrieved_memory,
    )
    interaction = {
        "task": task.model_dump(),
        "memory_system": memory_name,
        "llm_provider": llm_provider,
        "model_name": model_name,
        "temperature": temperature,
        "agent_response": result.response,
        "evaluation_result": {
            "task_success": eval_res.task_success,
            "rule_compliance_score": eval_res.rule_compliance_score,
            "violated_rules": eval_res.violated_rules,
            "correct_subtasks": eval_res.correct_subtasks,
            "total_subtasks": eval_res.total_subtasks,
            "progress_score": eval_res.progress_score,
            "memory_usage_quality": eval_res.memory_usage_quality,
        },
        "feedback": eval_res.feedback,
        "retrieved_memory": result.memory_as_dicts(),
        "token_usage": {
            "input_tokens": getattr(result.llm_response, "input_tokens", 0),
            "output_tokens": getattr(result.llm_response, "output_tokens", 0),
            "latency_s": getattr(result.llm_response, "latency_s", 0.0),
            "estimated_cost_usd": getattr(result.llm_response, "estimated_cost_usd", 0.0),
        },
    }
    if output_path:
        JsonlWriter(output_path).write(interaction)
    click.echo(json.dumps(interaction, indent=2, default=str))


@cli.command("analyze", help="Aggregate raw JSONL logs into CSV summaries + plots.")
@click.option("--results", "results_dir", default="results/raw_logs/",
              type=click.Path(file_okay=False), show_default=True)
@click.option("--output", "output_dir", default="results/", type=click.Path(file_okay=False),
              show_default=True)
@click.option("--no-plots", is_flag=True, default=False, help="Skip plot generation.")
def cmd_analyze(results_dir: str, output_dir: str, no_plots: bool) -> None:
    from .analysis.aggregate_results import aggregate
    from .analysis.statistical_analysis import run_basic_stats
    from .analysis.visualizations import render_all

    summary = aggregate(results_dir, output_dir=output_dir)
    stats = run_basic_stats(summary["frame_path"], output_dir=output_dir)
    if not no_plots:
        render_all(summary["frame_path"], output_dir=os.path.join(output_dir, "plots"))
    click.echo(json.dumps({"aggregate": summary, "stats": stats}, indent=2, default=str))


def _safe_load(path: str | Path) -> dict[str, Any]:
    try:
        return load_config(path)
    except FileNotFoundError:
        return {}


def main() -> None:  # pragma: no cover - thin wrapper
    cli()


if __name__ == "__main__":
    sys.exit(main())

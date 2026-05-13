# memory-agent-eval

A research platform for evaluating memory mechanisms in LLM-based agents on
iterative, rule-based **Design-for-X (DFx)** tasks.

The platform lets you run the *same* DFx scenarios across:

1. different LLM providers (OpenAI, Anthropic, Google, local Ollama, mock),
2. different memory systems (stateless, hindsight, contextual, persistent,
   plus pluggable wrappers for Mem0 / Zep / Supermemory / AContext),
3. multiple sessions per scenario (iteration depth),
4. repeated runs per condition (variance / reproducibility),
5. a unified evaluation pipeline (rule compliance, progress, latency, cost,
   memory utility).

It targets four research questions:

- **RQ1** how do different memory mechanisms influence agent performance and
  learning behaviour in iterative, interdependent tasks?
- **RQ2** to what extent do memory-augmented agents improve over iterations
  and transfer learned knowledge across related tasks vs stateless agents?
- **RQ3** how does the choice of LLM affect memory-based learning?
- **RQ4** how effectively do agents retrieve and utilise stored memory
  during task execution?

---

## Project layout

```
memory-agent-eval/
├── configs/                # experiment / model / memory YAML configs
├── data/
│   ├── tasks/              # multi-session DFx tasks (YAML)
│   ├── rules/              # DFM / DFA rule packs (YAML)
│   └── examples/           # canonical agent + feedback shapes
├── src/
│   ├── main.py             # CLI entry point
│   ├── experiment_runner.py
│   ├── agents/
│   ├── llms/
│   ├── memory/
│   ├── tasks/
│   ├── evaluation/
│   ├── analysis/
│   └── utils/
├── results/                # JSONL logs, CSV summaries, plots
├── tests/                  # pytest suite
├── dashboard/app.py        # optional Streamlit dashboard
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick start

```bash
cd memory-agent-eval

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # fill in API keys later if you want real LLMs

# 1. Smoke test on a single task with the deterministic Mock LLM.
python -m src.main run-single \
  --memory hindsight \
  --llm mock \
  --model mock-deterministic \
  --task data/tasks/enclosure_dfm_session_1.yaml

# 2. Full experiment (mock LLM, stateless vs hindsight vs contextual vs persistent).
python -m src.main run --config configs/experiment.yaml

# 3. Aggregate + plot.
python -m src.main analyze --results results/raw_logs/

# 4. (Optional) browse results in Streamlit.
streamlit run dashboard/app.py
```

The default config runs entirely offline using the **Mock LLM**, so you can
verify the pipeline before spending API credits. Uncomment provider entries
in `configs/experiment.yaml` once your `.env` is populated.

---

## Configuration

Three YAML files drive a run:

- `configs/experiment.yaml` - top-level: memory_systems, llms, tasks, runs.
- `configs/models.yaml` - per-provider defaults (timeouts, cost tables).
- `configs/memory_systems.yaml` - per-memory hyperparameters (top-k, paths).

All three are merged on top of platform defaults defined in
`src/experiment_runner.py`.

### Environment variables

```
# LLMs
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434

# External memory backends (self-hosted defaults)
MEM0_VECTOR_STORE=qdrant
MEM0_QDRANT_URL=http://localhost:6333
MEM0_LLM_PROVIDER=openai
MEM0_LLM_MODEL=gpt-4o-mini

ZEP_API_URL=http://localhost:8000
ZEP_API_KEY=

ACONTEXT_API_URL=http://localhost:8029/api/v1
ACONTEXT_API_KEY=sk-ac-your-root-api-bearer-token

HINDSIGHT_API_URL=http://localhost:8888
HINDSIGHT_API_KEY=
```

If any LLM key or memory backend is unreachable the corresponding class
falls back to a clearly-logged stub - the pipeline still runs.

---

## Memory systems

| Name           | Class                       | Backend                                          |
|----------------|-----------------------------|--------------------------------------------------|
| `stateless`    | `StatelessMemory`           | No-op baseline. Stores nothing.                  |
| `reflection`   | `ReflectionMemory`          | In-RAM hand-rolled reflections + Jaccard retrieval. (Was `hindsight` in v0.1.) |
| `contextual`   | `ContextualMemory`          | In-RAM richer per-task context store.            |
| `persistent`   | `PersistentMemory`          | JSONL on disk, survives restarts.                |
| `mem0`         | `Mem0Memory`                | **Self-hosted Mem0** via Qdrant + an LLM (in-process library). |
| `zep`          | `ZepMemory`                 | **Self-hosted Zep Community Edition** via Docker compose. |
| `acontext`     | `AContextMemory`            | **Self-hosted AContext** via `acontext server up`. |
| `hindsight`    | `VectorizeHindsightMemory`  | **Self-hosted Vectorize.io Hindsight** via Docker. |

All implement the same interface:

```python
class BaseMemory:
    def retrieve(self, query: str, context: dict) -> list[MemoryItem]: ...
    def update(self, interaction: dict) -> None: ...
    def reset(self) -> None: ...
    def export_memory(self) -> dict: ...
```

> **Naming note.** In v0.1 the `hindsight` key referred to our hand-rolled
> reflection technique. From v0.2 onwards `hindsight` refers to Vectorize.io's
> Hindsight memory service, and the old technique was renamed to `reflection`.
> A backwards-compatibility shim still lets `from src.memory.hindsight_memory
> import HindsightMemory` work; it emits a `DeprecationWarning` and behaves
> exactly like `ReflectionMemory`.

> **Supermemory** is intentionally not registered: it has no fully
> self-hostable deployment today. To re-enable, see
> `src/memory/supermemory_memory.py`.

### External memory configuration

External memories take a `reset_policy` kwarg (set per-memory in
`configs/memory_systems.yaml`):

| Policy             | Local mirror | Remote data | Use case                                  |
|--------------------|--------------|-------------|-------------------------------------------|
| `clear_remote`     | cleared      | cleared     | Default - per-run isolation for stats.    |
| `keep_remote`      | cleared      | kept        | Study cross-run / long-term learning.     |
| `clear_local_only` | cleared      | kept        | Explicit alias of `keep_remote`.          |

Adding a new memory only requires implementing the four `BaseMemory`
methods and registering the class in `src/memory/__init__.py::_REGISTRY`.

### Infrastructure setup (self-hosted)

Bring up only the servers you intend to compare. All four wrappers fall
back to a clearly-logged in-RAM stub if their server is unreachable, so
this is fully incremental.

**1. Qdrant (vector store for Mem0)**

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
pip install mem0ai qdrant-client
```

**2. Zep Community Edition**

```bash
git clone https://github.com/getzep/zep && cd zep
docker compose up -d
pip install zep-cloud
```

**3. AContext**

```bash
curl -fsSL https://install.acontext.io | sh
mkdir -p acontext_server && cd acontext_server
echo 'LLM_API_KEY="$OPENAI_API_KEY"' > .env
acontext server up
pip install acontext
```

**4. Vectorize Hindsight**

```bash
docker run -d --name hindsight -p 8888:8888 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  ghcr.io/vectorize-io/hindsight
pip install hindsight-client
```

**Verify all four are healthy:**

```bash
for url in http://localhost:8888/healthz http://localhost:6333/collections \
           http://localhost:8000/healthz http://localhost:8029/api/v1/healthz; do
  echo "── $url"; curl -sS -o /dev/null -w "%{http_code}\n" $url
done
```

Then uncomment the corresponding entries in `configs/experiment.yaml::memory_systems`.

---

## LLM providers

| Name        | Provider                | Auth                  |
|-------------|-------------------------|-----------------------|
| `openai`    | OpenAI Chat Completions | `OPENAI_API_KEY`      |
| `anthropic` | Anthropic Claude        | `ANTHROPIC_API_KEY`   |
| `google`    | Google Gemini           | `GOOGLE_API_KEY`      |
| `local`     | Ollama HTTP             | `OLLAMA_BASE_URL`     |
| `mock`      | Deterministic stub      | none (offline / CI)   |

All implement:

```python
class BaseLLM:
    def generate(self, messages: list[dict], temperature: float = 0.0) -> LLMResponse: ...
```

`LLMResponse` carries text + token counts + latency + estimated USD cost,
so the metrics module can produce efficiency tables without provider-specific
code.

---

## Task design

Each `data/tasks/*.yaml` file is one **session** of a multi-session
**scenario**. The runner groups files by `scenario_name` and replays them in
order of `session_id` so memory-dependent agents see the natural progression.

Required fields:

```yaml
task_id: enclosure_dfm_s1
scenario_name: handheld_enclosure
session_id: 1
input_description: ...
design_context: { wall_thickness_mm: 1.1, ... }
dfx_rules_path: data/rules/dfm_rules.yaml
constraints: [ "Cycle time <= 35s", ... ]
expected_output_format: structured_json
hidden_dependency_from_previous_sessions: ...
evaluation_criteria:
  rule_pack: dfm
  ground_truth_design: { ... }
  expected_violations: [DFM-001, ...]
  required_fields: [summary, dfx_rule_analysis, final_recommendation]
  total_subtasks: 5
```

The bundled scenario `handheld_enclosure` runs four sessions: a DFM review
(s1), a partial fix (s2), a DFM optimisation under a new constraint (s3),
and a transfer task to DFA rules (s4).

---

## Agent output schema

Every agent produces a single JSON object with this shape (parsed
tolerantly - markdown fences and stray prose are stripped):

```json
{
  "summary": "...",
  "decision": "...",
  "dfx_rule_analysis": [
    {"rule_id": "...", "status": "satisfied|violated|uncertain", "explanation": "..."}
  ],
  "used_memory": [{"memory_id": "...", "how_used": "..."}],
  "final_recommendation": "...",
  "confidence": 0.0
}
```

## Feedback schema

The evaluator returns canonical structured feedback that the next session's
agent receives via `<PRIOR_FEEDBACK>` (and that the memory module ingests):

```json
{
  "task_success": true,
  "rule_compliance_score": 0.85,
  "violated_rules": [],
  "correct_subtasks": 4,
  "total_subtasks": 5,
  "feedback_summary": "...",
  "improvement_suggestions": [...],
  "memory_usage_quality": {
    "retrieval_relevance": 0.8,
    "retrieval_correctness": 0.9,
    "usage_quality": 0.75
  }
}
```

---

## Evaluation metrics

Implemented per-interaction in `src/evaluation/`:

| Metric                  | Where computed                                |
|-------------------------|-----------------------------------------------|
| Task success            | `MetricsCalculator.task_success`              |
| Rule compliance score   | `MetricsCalculator.rule_compliance`           |
| Progress score          | `MetricsCalculator.progress_score`            |
| Memory utility (3 sub-metrics) | `MetricsCalculator.memory_utility`     |
| Latency / tokens / cost | Carried in `LLMResponse`, logged per record   |

Cross-run metrics computed in `src/analysis/`:

| Metric                  | Where computed                                |
|-------------------------|-----------------------------------------------|
| Improvement-over-time slope | `aggregate_results.aggregate`             |
| Consistency (std dev)   | `aggregate_results.aggregate` (`std_compliance`) |
| Stateless-baseline lift | `aggregate_results.aggregate`                 |
| ANOVA + Tukey + Cohen's d | `statistical_analysis.run_basic_stats`      |

---

## Logging

Every interaction is one JSONL line. Each record contains:

`experiment_id`, `run_id`, `task_id`, `session_id`, `memory_system`,
`llm_provider`, `model_name`, `temperature`, `seed`, `prompt`,
`retrieved_memory`, `agent_response`, `evaluation_result`, `feedback`,
`memory_update`, `latency_s`, `token_usage`, `timestamp`.

Files land under `results/raw_logs/<experiment_id>.jsonl`.

---

## Analysis & plots

```bash
python -m src.main analyze --results results/raw_logs/
```

Produces:

- `results/metrics/interactions.csv` (flat long-form frame)
- `results/metrics/by_condition.csv`
- `results/metrics/by_iteration.csv`
- `results/metrics/improvement_slope.csv`
- `results/metrics/stateless_lift.csv`
- `results/metrics/stats.json` (ANOVA / Tukey / effect sizes)
- `results/plots/*.png`:
  - `success_by_memory.png` (RQ1)
  - `compliance_by_iteration.png` (RQ2)
  - `memory_utility.png` (RQ4)
  - `model_comparison.png` (RQ3)
  - `stateless_vs_augmented.png` (RQ2)
  - `improvement_slope.png` (RQ2)

---

## Tests

```bash
pytest -q
```

Covers the rule checker, every memory implementation, the evaluator, and a
full pipeline smoke test (stateless vs hindsight on the canonical scenario).

---

## Reproducibility notes

- `temperature` defaults to 0.0.
- `seed` is logged on every record but currently advisory: the deterministic
  components (rule checker, mock LLM) are exact, while real LLMs remain the
  only stochastic source.
- Memory is reset between repeated runs of the same condition; set
  `cross_scenario_memory: true` in the experiment config to test transfer.
- The stateless agent uses the *same prompt template* and *same evaluator*
  as memory-augmented agents - the only difference is the memory module.

---

## Roadmap

- ✅ Real Mem0, Zep CE, AContext, and Vectorize Hindsight integrations
  (self-hosted; SDKs are optional dependencies).
- ✅ Configurable `reset_policy` per external memory.
- ✅ `run_id` propagation to external memories so per-run session/bank
  scoping works cleanly.
- Supermemory integration once their self-host story is complete.
- Embedding-based retrieval for `ContextualMemory`, `PersistentMemory`,
  and `ReflectionMemory`.
- LLM-based reflection summariser for `ReflectionMemory`
  (the `use_llm_summarizer` flag is already plumbed through).
- Dedicated dashboard "Memory inspector" page (retrieval distribution,
  growth-over-time, retrieved-vs-used Sankey).

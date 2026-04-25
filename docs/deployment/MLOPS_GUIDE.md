# MLOps Guide

Evaluation, data versioning, and experiment tracking for MedExRAG. For DVC and MLflow fundamentals see the [DVC docs](https://dvc.org/doc) and [MLflow docs](https://mlflow.org/docs/latest/).

## DVC pipeline

Defined in `dvc.yaml`. Two stages:

| Stage | Command | Deps | Outs / Metrics |
|-------|---------|------|----------------|
| `ingest` | Builds ChromaDB from PDFs in `data/medical_literature/` | `data/medical_literature/`, `src/medexrag/pipeline.py` | `data/medical_kb/` |
| `evaluate` | Runs retrieval + E2E + LLM-judge metrics | `src/medexrag/evaluation/`, `data/medical_kb/` | `metrics/eval_metrics.json` |

### Run

```bash
PYTHONPATH=src dvc repro             # full pipeline
PYTHONPATH=src dvc repro evaluate    # one stage
dvc dag                              # show DAG
dvc metrics show                     # display metrics/eval_metrics.json
dvc metrics diff HEAD~1              # compare to previous commit
```

DVC stages set `sys.path.insert(0, 'src')` inline so they work cross-platform without `PYTHONPATH`.

## DVC remote

Config lives in `.dvc/config`. The repo currently has commented examples for S3/GCS/Azure. Pick one and configure:

```bash
# S3
dvc remote add -d storage s3://your-bucket/medexrag-dvc
dvc remote modify storage region us-east-1

# GCS
dvc remote add -d storage gs://your-bucket/medexrag-dvc

# Azure
dvc remote add -d storage azure://container/medexrag-dvc
```

Push/pull tracked artifacts:

```bash
dvc add data/medical_kb/   # track
dvc push                   # upload to remote
dvc pull                   # download
```

Credentials in CI come from the secrets listed in `docs/deployment/CICD_GUIDE.md`.

## Metrics directory

`metrics/eval_metrics.json` is the canonical output, written by the `evaluate` stage. Tracked by DVC, displayed via `dvc metrics show`, and read by the `quality-gate` job in `.github/workflows/mlops.yml`.

Thresholds enforced in `tests/test_evaluation.py::TestQualityGates`:

| Metric | Threshold |
|--------|-----------|
| E2E pass rate | >= 0.70 |
| Precision@5 | >= 0.30 |
| MRR | >= 0.50 |
| LLM-judge overall | >= 0.40 |

## Evaluation modules

In `src/medexrag/evaluation/`:

- `retrieval_metrics.py` — Precision@k, Recall@k, MRR, NDCG@k, Hit-Rate@k
- `agent_evaluators.py` — Per-agent quality (Analyst, Researcher, Diagnostician, Reporter)
- `e2e_benchmarks.py` — End-to-end test cases (categories: `functional`, `performance`, `robustness`)
- `llm_judge.py` — LLM-as-judge scoring (faithfulness, relevance, coherence, clinical appropriateness, citation accuracy, completeness). Mock mode for CI.
- `mlflow_tracking.py` — `MLflowTracker` wrapper

## MLflow tracking

```python
from medexrag.evaluation.mlflow_tracking import MLflowTracker

tracker = MLflowTracker(
    experiment_name="medical-rag-evaluation",
    tracking_uri="http://localhost:5000",   # or set MLFLOW_TRACKING_URI
)

with tracker.start_run(run_name="rag-v1.0"):
    tracker.log_rag_config({"model_name": "Qwen2-VL-2B", "k_sources": 5})
    tracker.log_retrieval_metrics(retrieval_metrics)
    tracker.log_agent_metrics(agent_results)
    tracker.log_e2e_metrics(benchmark_results)
```

Local UI: `mlflow ui --port 5000`.

In CI, set `MLFLOW_TRACKING_URI` repository secret to point at a hosted server, otherwise the file backend is used and runs are ephemeral.

## File map

```
dvc.yaml                              Pipeline definition
dvc.lock                              Resolved dependency hashes
.dvc/config                           Remote config
metrics/eval_metrics.json             Evaluation output
src/medexrag/evaluation/              Evaluation code
tests/test_evaluation.py              Quality gate tests
.github/workflows/mlops.yml           CI integration
```

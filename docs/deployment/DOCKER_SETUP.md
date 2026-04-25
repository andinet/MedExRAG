# Docker Setup

Containerized deployment for MedExRAG. For Docker fundamentals see the [official docs](https://docs.docker.com/).

## Files

| File | Purpose |
|------|---------|
| `docker/docker-compose.yml` | Service orchestration (CPU baseline) |
| `docker/docker-compose.gpu.yml` | GPU overlay (NVIDIA runtime) |
| `docker/Dockerfile.base` | Python 3.11 + ML deps (torch, transformers, chromadb) |
| `docker/Dockerfile.streamlit` | Streamlit web service |
| `docker/Dockerfile.cli` | CLI worker for batch jobs |
| `docker/Dockerfile.vlm` | Standalone VLM serving image |

## Services

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| `streamlit` | `Dockerfile.streamlit` | 8501, 8002 | Web UI + Prometheus metrics |
| `cli-worker` | `Dockerfile.cli` | - | Batch jobs (run with `--rm`) |
| `prometheus` | `prom/prometheus` | 9090 | Metrics scrape (config `config/observability/prometheus.yml`) |
| `grafana` | `grafana/grafana` | 3000 | Dashboards (default `admin`/`admin`) |
| `jaeger` | `jaegertracing/all-in-one` | 16686 | Distributed traces |
| `otel-collector` | `otel/opentelemetry-collector` | 4317 | OTLP ingest (config `config/observability/otel-collector-config.yaml`) |

## Volumes

Host bind mounts: `data/medical_kb/` (ChromaDB), `data/medical_literature/` (PDF inputs), `data/xray_images/` (inputs), `data/analysis_results/` (outputs).

Named volumes: `huggingface_cache` (~5GB model cache), `prometheus_data`, `grafana_data`.

In-container paths are `/app/data/...`.

## Start the stack

```bash
# Web UI + full observability (CPU)
docker compose -f docker/docker-compose.yml up streamlit prometheus grafana jaeger otel-collector

# With GPU (requires NVIDIA Container Toolkit, or Docker Desktop + WSL2 on Windows)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up streamlit prometheus grafana jaeger otel-collector

# Detached
docker compose -f docker/docker-compose.yml up -d streamlit

# Stop
docker compose -f docker/docker-compose.yml down
```

Access: Streamlit `http://localhost:8501`, Grafana `http://localhost:3000`, Prometheus `http://localhost:9090`, Jaeger `http://localhost:16686`, raw metrics `http://localhost:8002/metrics`.

## CLI worker patterns

The `cli-worker` service is one-shot. Always invoke with `run --rm`:

```bash
# Ingest PDFs from data/medical_literature/
docker compose -f docker/docker-compose.yml run --rm cli-worker \
  ingest /app/data/medical_literature

# Single-image analysis with RAG
docker compose -f docker/docker-compose.yml run --rm cli-worker \
  analyze /app/data/xray_images/chest.dcm --question "Analyze for pneumonia" --rag

# Batch analysis
docker compose -f docker/docker-compose.yml run --rm cli-worker \
  batch /app/data/xray_images --output /app/data/analysis_results --rag --workers 4

# Literature search
docker compose -f docker/docker-compose.yml run --rm cli-worker \
  search "pneumonia consolidation" --k 5

# KB stats
docker compose -f docker/docker-compose.yml run --rm cli-worker stats
```

## Build

```bash
docker compose -f docker/docker-compose.yml build              # all
docker compose -f docker/docker-compose.yml build streamlit    # one
docker compose -f docker/docker-compose.yml build --no-cache   # clean
```

## Environment variables

Set in `docker/docker-compose.yml` under each service's `environment:` block. See project `CLAUDE.md` for the full list. Notable ones:

- `MEDICAL_KB_PATH` (default `/app/data/medical_kb`)
- `LITERATURE_DIR` (default `/app/data/medical_literature`)
- `MODEL_NAME` (default `Qwen/Qwen2-VL-2B-Instruct` — override to `Qwen/Qwen2-VL-0.5B-Instruct` if memory-constrained)
- `RAG_K_SOURCES` (default `5`)
- `LANGCHAIN_API_KEY` (optional, enables LangSmith)

## Common issues

- **Exit 137**: container OOM-killed. Raise Docker Desktop memory to 12GB+ or switch to the 0.5B model.
- **Port already in use**: change the host-side port mapping in `docker/docker-compose.yml`.
- **Model download stalls**: pre-warm `~/.cache/huggingface` on the host and bind-mount it; verify the container has internet (`curl -I https://huggingface.co`).
- **GPU not detected**: confirm with `docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml run --rm cli-worker python -c "import torch; print(torch.cuda.is_available())"`.

For Kubernetes/EKS deployment see `infra/k8s/` and `docs/deployment/CICD_GUIDE.md`.

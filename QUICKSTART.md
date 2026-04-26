# MedExRAG Quick Reference

## First-Time Setup

The `streamlit` and `cli-worker` services extend a local `medexrag-base:latest` image that is not pulled from a registry. Build it once before the first `up`:

```bash
# Run from the repo root
docker build -t medexrag-base:latest -f docker/Dockerfile.base .
```

This downloads PyTorch + transformers and takes several minutes. After it succeeds, continue with "Start Everything" below.

## Start Everything

```bash
# All services (GPU)
DOCKER_API_VERSION=1.44 docker-compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d streamlit prometheus grafana jaeger otel-collector

# All services (CPU only)
DOCKER_API_VERSION=1.44 docker-compose -f docker/docker-compose.yml up -d streamlit prometheus grafana jaeger otel-collector
```

## Stop Everything

```bash
DOCKER_API_VERSION=1.44 docker-compose -f docker/docker-compose.yml down
```

## Rebuild After Code Changes

```bash
# IMPORTANT: must rebuild base image first (it contains the source code)
docker build -t medexrag-base:latest -f docker/Dockerfile.base .
docker build -t medexrag-streamlit:latest -f docker/Dockerfile.streamlit .

# Restart only streamlit (other services keep running)
docker stop medexrag-streamlit && docker rm medexrag-streamlit
DOCKER_API_VERSION=1.44 docker-compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d --no-deps streamlit
```

## Service URLs

| Service | URL |
|---------|-----|
| Web UI | http://localhost:8501 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| Jaeger | http://localhost:16686 |
| Metrics | http://localhost:8002/metrics |

## Check Status

```bash
# All containers
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' | grep medexrag

# Streamlit logs
docker logs medexrag-streamlit --tail 30

# Follow logs live
docker logs -f medexrag-streamlit

# Shell into container
docker exec -it medexrag-streamlit /bin/bash
```

## CLI Commands

```bash
# Ingest literature
DOCKER_API_VERSION=1.44 docker-compose -f docker/docker-compose.yml run --rm cli-worker ingest /app/data/medical_literature

# Analyze single X-ray
DOCKER_API_VERSION=1.44 docker-compose -f docker/docker-compose.yml run --rm cli-worker analyze /app/data/xray_images/chest.dcm --rag

# Search literature
DOCKER_API_VERSION=1.44 docker-compose -f docker/docker-compose.yml run --rm cli-worker search "pneumonia consolidation" --k 5

# Batch process
DOCKER_API_VERSION=1.44 docker-compose -f docker/docker-compose.yml run --rm cli-worker batch /app/data/xray_images --output /app/data/analysis_results --rag
```

## Local Dev (no Docker)

```bash
# Run Streamlit
PYTHONPATH=src streamlit run src/medexrag/app.py

# Run tests
PYTHONPATH=src pytest tests/ -v

# Test pipeline import
PYTHONPATH=src python -c "from medexrag.pipeline import MedicalRAGPipeline; p = MedicalRAGPipeline(load_vlm=False); print(p.get_stats())"
```

## Quick Debugging

```bash
# Port conflict? Find what's using it
ss -tlnp | grep <port>
docker ps --format '{{.Names}}\t{{.Ports}}' | grep <port>

# Container running but unreachable? Check port bindings
docker port medexrag-streamlit

# Check VLM latency
curl -s http://localhost:8002/metrics | grep "vlm_inference_seconds_sum"

# Check knowledge base size
curl -s http://localhost:8002/metrics | grep "vector_store_chunks"

# Find slow traces in Jaeger
curl -s "http://localhost:16686/api/traces?service=medical-rag&lookback=1h&minDuration=15s"

# Filter logs by request
docker logs medexrag-streamlit 2>&1 | grep "<correlation_id>"
```

## Data Directories

| Path | Contents |
|------|----------|
| `data/medical_literature/` | Place PDFs here for ingestion |
| `data/medical_kb/` | ChromaDB vector database (auto-generated) |
| `data/xray_images/` | X-ray images for analysis |
| `data/analysis_results/` | Output from batch processing |

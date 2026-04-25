# MedExRAG Documentation

Index of documentation for the MedExRAG agentic X-ray RAG system. See the project [README](../README.md) for setup.

## Architecture
- [RAG Architecture](architecture/RAG_ARCHITECTURE.md) - data flow, components, two-stage RAG, multi-agent layer

## Guides
- [RAG Guide](guides/RAG_GUIDE.md) - ingestion, chunking, retrieval
- [VLM Guide](guides/VLM_GUIDE.md) - model selection, quantization, VRAM
- [Agents Guide](guides/AGENTS_GUIDE.md) - LangGraph state machine and tools
- [Streamlit Guide](guides/STREAMLIT_GUIDE.md) - web UI tabs and workflows

## Deployment
- [Docker Setup](deployment/DOCKER_SETUP.md) - compose, GPU overlay, volumes
- [CI/CD Guide](deployment/CICD_GUIDE.md) - GitHub Actions workflows
- [MLOps Guide](deployment/MLOPS_GUIDE.md) - DVC pipeline, evaluation, EKS

## Observability
- [Observability](observability/README.md) - Prometheus, Grafana, Jaeger, OTel, LangSmith

## Operations
- [Troubleshooting](TROUBLESHOOTING.md) - common issues and fixes

## Service URLs (local Docker)

| Service | URL |
|---------|-----|
| Streamlit | http://localhost:8501 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Jaeger | http://localhost:16686 |
| Metrics | http://localhost:8002/metrics |

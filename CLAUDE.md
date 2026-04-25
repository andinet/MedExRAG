# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project does

MedExRAG is an agentic RAG system for X-ray interpretation. A vision-language model (Qwen2-VL by default) analyzes the image, relevant medical literature is retrieved from a ChromaDB vector store with PubMedBERT embeddings, and a LangGraph multi-agent workflow (analyst → researcher → diagnostician → reporter) produces an evidence-grounded report. Prometheus + Grafana provide observability.

## Layout

- `src/medexrag/` — main package: `pipeline.py`, `agents.py`, `app.py` (Streamlit), `cli.py`, `observability/`, `evaluation/`
- `tests/` — pytest suite
- `docker/`, `infra/`, `config/` — Docker, Kubernetes/EKS, Prometheus/Grafana/OTel config
- `examples/` — example scripts
- `docs/` — guides

## Run

```bash
pip install -e .

# Web UI
streamlit run src/medexrag/app.py

# CLI
python -m medexrag.cli analyze <image> --rag
python -m medexrag.cli ingest data/medical_literature/

# Tests
pytest tests/ -v
```

Docker: `docker compose -f docker/docker-compose.yml up streamlit`

## Notes for Claude

- Python 3.11+, package uses `src/` layout (already configured in `pyproject.toml`).
- Default VLM is `Qwen/Qwen2-VL-2B-Instruct`; 4-bit quantization auto-enables on GPUs with <8 GB VRAM.
- Pass `load_vlm=False` to `MedicalRAGPipeline()` for ingestion-only work.
- Observability stack runs on standard ports: Streamlit 8501, Grafana 3000, Prometheus 9090, Jaeger 16686.

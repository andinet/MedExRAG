# MedExRAG — Medical Expert X-ray Analysis with RAG

A production-ready system combining **Vision Language Models (VLMs)** with **Retrieval-Augmented Generation (RAG)** for evidence-based radiology interpretation, powered by a medical literature knowledge base and multi-agent LangGraph orchestration.

## Features

- **Multi-Agent Orchestration**: LangGraph state machine with four specialized agents (analyst → researcher → diagnostician → reporter) and tool-calling via LangChain
- **Two-Stage RAG**: Initial VLM analysis seeds a literature query against a ChromaDB vector store of medical literature embedded with PubMedBERT, then re-prompts the VLM with the retrieved evidence
- **Full Observability Stack**: LangSmith (LLM tracing), Prometheus (metrics), OpenTelemetry/Jaeger (distributed tracing), Grafana (dashboards + alerts)
- **Production Deployment**: Docker Compose for local, Kubernetes/EKS via Terraform for cloud, with Karpenter scale-to-zero GPU node pools
- **MLOps**: DVC for reproducible pipelines (ingest, evaluate), MLflow for experiment tracking, GitHub Actions CI/CD with OIDC to AWS
- **Multiple Interfaces**: Streamlit web UI (chat, upload, literature, batch), CLI, and programmatic Python API
- **PDF Ingestion**: DocLing extracts text and tables from medical PDFs with OCR support for scanned documents

## Quick Start

Choose either **Docker** (recommended for production) or **Local Setup** (for development).

### Option A: Docker Setup (Recommended)

#### Prerequisites
- Docker 20.10+ and Docker Compose 2.0+
- 12GB+ RAM allocated to Docker
- 15GB+ free disk space
- **For GPU**: NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

#### Start the Application (GPU)

```bash
# Clone repository
git clone <your-repo-url>
cd MedExRAG

# Copy environment template and configure
cp .env.example .env

# Build and start services with GPU support + full observability
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up streamlit prometheus grafana jaeger otel-collector

# Access:
#   Web UI:      http://localhost:8501
#   Grafana:     http://localhost:3000 (admin/admin)
#   Prometheus:  http://localhost:9090
#   Jaeger:      http://localhost:16686
#   Metrics:     http://localhost:8002/metrics
```

#### Start the Application (CPU Fallback)

```bash
docker compose -f docker/docker-compose.yml up streamlit prometheus grafana jaeger otel-collector
```

#### Common Docker Commands

```bash
# View logs
docker compose -f docker/docker-compose.yml logs -f streamlit

# Ingest medical literature
docker compose -f docker/docker-compose.yml run --rm cli-worker ingest /app/data/medical_literature

# Analyze a single X-ray
docker compose -f docker/docker-compose.yml run --rm cli-worker analyze /app/data/xray_images/chest.dcm --rag

# Batch process X-rays
docker compose -f docker/docker-compose.yml run --rm cli-worker batch /app/data/xray_images --output /app/data/analysis_results --rag

# Search literature
docker compose -f docker/docker-compose.yml run --rm cli-worker search "pneumonia consolidation" --k 5

# Get knowledge base statistics
docker compose -f docker/docker-compose.yml run --rm cli-worker stats

# Stop all services
docker compose -f docker/docker-compose.yml down
```

---

### Option B: Local Setup (Development)

#### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd MedExRAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  (Windows)

# Install dependencies
pip install -r requirements.txt

# Run setup script (creates directories, tests imports)
python setup.py
```

#### 2. Prepare Medical Literature (Optional)

Add medical PDF files to `data/medical_literature/`, then ingest them:

```bash
PYTHONPATH=src python examples/test_ingestion.py
```

#### 3. Run the Application

```bash
PYTHONPATH=src streamlit run src/medexrag/app.py
```

The app opens at `http://localhost:8501`.

#### 4. CLI Interface

```bash
# Analyze single X-ray with RAG
PYTHONPATH=src python -m medexrag.cli analyze chest_xray.dcm --rag

# Ingest literature
PYTHONPATH=src python -m medexrag.cli ingest data/medical_literature/

# Search literature
PYTHONPATH=src python -m medexrag.cli search "pneumonia consolidation" --k 5

# Batch processing
PYTHONPATH=src python -m medexrag.cli batch data/xray_images/ --output data/analysis_results/ --rag
```

## Architecture

```
┌──────────────────┐
│   Streamlit UI   │
└────────┬─────────┘
         │
         v
┌──────────────────────────────────┐
│   MedicalRAGPipeline             │
│   ┌──────────────────────────┐   │
│   │ DocLingProcessor         │   │  ← PDF ingestion
│   │ (Medical Literature)     │   │
│   └────────────┬─────────────┘   │
│                v                 │
│   ┌──────────────────────────┐   │
│   │ MedicalVectorStore       │   │  ← ChromaDB + PubMedBERT
│   │ (Semantic Search)        │   │
│   └────────────┬─────────────┘   │
│                v                 │
│   ┌──────────────────────────┐   │
│   │ VLMInference             │   │  ← Qwen2-VL-2B-Instruct
│   │ (Image Analysis)         │   │
│   └──────────────────────────┘   │
└──────────────────────────────────┘
```

### RAG Workflow

1. **Initial VLM Analysis**: VLM analyzes X-ray image → generates preliminary findings
2. **Literature Retrieval**: Findings used as search query → retrieves relevant medical literature from ChromaDB
3. **Enhanced Analysis**: VLM receives image + findings + literature → produces evidence-based analysis with citations

### Multi-Agent System

The LangGraph-based agent architecture (`src/medexrag/agents.py`) provides:

- **Analyst Agent**: Performs initial X-ray image analysis via VLM
- **Researcher Agent**: Searches medical literature for supporting evidence
- **Diagnostician Agent**: Synthesizes findings with literature into differential diagnoses
- **Reporter Agent**: Generates structured clinical reports with citations

## Project Structure

```
MedExRAG/
├── README.md                        # This file
├── CLAUDE.md                        # Claude Code development instructions
├── LICENSE                          # Apache 2.0 License
├── pyproject.toml                   # Python project configuration
├── requirements.txt                 # Python dependencies
├── setup.py                         # Setup script
├── .env.example                     # Environment template
├── dvc.yaml                         # DVC pipeline definitions
│
├── src/medexrag/                    # Main Python package
│   ├── __init__.py                  # Package init + version
│   ├── __main__.py                  # python -m medexrag entry
│   ├── pipeline.py                  # RAG pipeline core (DocLing + ChromaDB + VLM)
│   ├── agents.py                    # LangGraph multi-agent system
│   ├── app.py                       # Streamlit web application
│   ├── cli.py                       # Command-line interface
│   ├── observability/               # Logging, metrics, tracing, health
│   └── evaluation/                  # Benchmarks, LLM judge, retrieval metrics
│
├── tests/                           # Pytest test suite
├── docker/                          # Docker deployment files
├── config/                          # Prometheus, Grafana, OTel config
├── docs/                            # All documentation (categorized)
├── examples/                        # Example usage scripts
├── data/                            # Runtime data (gitignored)
│   ├── medical_kb/                  # ChromaDB vector database
│   ├── medical_literature/          # Source PDF files
│   ├── xray_images/                 # X-ray images
│   └── analysis_results/            # Analysis outputs
└── metrics/                         # DVC evaluation metrics
```

## Observability

The system includes a comprehensive 4-milestone observability stack:

| Milestone | Component | Port | Purpose |
|-----------|-----------|------|---------|
| M1 | LangSmith | Cloud | LLM call tracing |
| M2 | Prometheus | 9090 | Performance metrics |
| M3 | Jaeger | 16686 | Distributed tracing |
| M4 | Grafana | 3000 | Dashboards & alerting |

```bash
# Start full observability stack
docker compose -f docker/docker-compose.yml up streamlit prometheus grafana jaeger otel-collector
```

See [docs/observability/](docs/observability/) for detailed guides.

## Configuration

### Change VLM Model

```python
from medexrag.pipeline import MedicalRAGPipeline

pipeline = MedicalRAGPipeline(
    model_name="microsoft/Phi-3-vision-128k-instruct",
    load_vlm=True
)
```

### Supported VLM Models
- `Qwen/Qwen2-VL-2B-Instruct` (default, 4GB)
- `Qwen/Qwen2-VL-0.5B-Instruct` (smaller, 1.5GB)
- `microsoft/Phi-3-vision-128k-instruct`
- `llava-hf/llava-1.5-7b-hf`

## Testing

```bash
# Run test suite
PYTHONPATH=src pytest tests/ -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=src/medexrag --cov-report=html

# Run evaluation pipeline (via DVC)
PYTHONPATH=src dvc repro evaluate
```

## Documentation

| Topic | Doc |
|---|---|
| Architecture | [RAG Architecture](docs/architecture/RAG_ARCHITECTURE.md) |
| Guides | [RAG](docs/guides/RAG_GUIDE.md) · [VLM](docs/guides/VLM_GUIDE.md) · [Agents](docs/guides/AGENTS_GUIDE.md) · [Streamlit](docs/guides/STREAMLIT_GUIDE.md) |
| Deployment | [Docker](docs/deployment/DOCKER_SETUP.md) · [CI/CD](docs/deployment/CICD_GUIDE.md) · [MLOps](docs/deployment/MLOPS_GUIDE.md) |
| Observability | [Prometheus / Grafana / Jaeger / LangSmith](docs/observability/README.md) |
| Operations | [Troubleshooting](docs/TROUBLESHOOTING.md) |

## Requirements

- Python 3.11+
- 8GB+ RAM (16GB recommended for VLM inference)
- GPU optional (CPU works but slower: ~10-30s/image vs ~2-5s on GPU)
- ~10GB disk space (VLM models + dependencies)

## Technology Stack

| Category | Technology |
|----------|-----------|
| Frontend | Streamlit |
| VLM | Qwen2-VL-2B-Instruct |
| Embeddings | PubMedBERT |
| Vector Store | ChromaDB |
| PDF Processing | DocLing (with OCR) |
| Agent Framework | LangChain + LangGraph |
| Observability | LangSmith, Prometheus, Jaeger, Grafana |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |

## License

Licensed under the [Apache License 2.0](LICENSE).

## Citation

```bibtex
@software{medexrag,
  title={MedExRAG: Medical Expert X-ray Analysis with RAG},
  author={Andinet Enquobahrie},
  year={2025},
  url={https://github.com/andinet/MedExRAG}
}
```

## Acknowledgments

- **DocLing**: PDF processing and document understanding
- **LangChain / LangGraph**: RAG framework and agent orchestration
- **Qwen-VL**: Vision Language Model by Alibaba
- **ChromaDB**: Vector database for semantic search
- **HuggingFace**: Model hosting and transformers library

---

**Need Help?** Check the [documentation](docs/) or open an issue!

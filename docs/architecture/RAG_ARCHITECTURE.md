# MedExRAG Architecture

Evidence-based radiology interpretation built on a Vision Language Model (VLM) grounded in a medical literature corpus.

## Data Flow

```
                          [PDFs in data/medical_literature/]
                                       |
                                       v
                       +-------------------------------+
                       |   DocLingProcessor            |
                       |   (OCR, tables, markdown)     |
                       +-------------------------------+
                                       |
                                       v
                       +-------------------------------+
                       |   RecursiveCharacterSplitter  |
                       |   chunk=800, overlap=150      |
                       +-------------------------------+
                                       |
                                       v
                       +-------------------------------+
                       |   ChromaDB + PubMedBERT       |
                       |   (persistent vector store)   |
                       +-------------------------------+
                                       ^
                                       | retrieve(top-k)
                                       |
   [X-ray image] --> [VLM: initial findings] --> [literature query] --> [VLM: grounded report]
                                                                              |
                                                                              v
                                                                    [report + citations]
```

## Components

| Component | File | Role |
|-----------|------|------|
| `DocLingProcessor` | [src/medexrag/pipeline.py](../../src/medexrag/pipeline.py) | Extracts text/tables from PDFs with OCR; emits markdown |
| `MedicalVectorStore` | [src/medexrag/pipeline.py](../../src/medexrag/pipeline.py) | ChromaDB persistent client; PubMedBERT embeddings; cosine similarity |
| `VLMInference` | [src/medexrag/pipeline.py](../../src/medexrag/pipeline.py) | Loads Qwen2-VL-2B-Instruct; auto 4-bit quantization on low-VRAM GPUs |
| `MedicalRAGPipeline` | [src/medexrag/pipeline.py](../../src/medexrag/pipeline.py) | Orchestrates ingestion, retrieval, and two-stage analysis |
| Multi-agent graph | [src/medexrag/agents.py](../../src/medexrag/agents.py) | LangGraph state machine (analyst -> researcher -> diagnostician -> reporter) |
| Streamlit UI | [src/medexrag/app.py](../../src/medexrag/app.py) | Chat, upload, literature management, batch tabs |
| CLI | [src/medexrag/cli.py](../../src/medexrag/cli.py) | `ingest`, `analyze`, `batch`, `search`, `stats` |
| Observability | [src/medexrag/observability/](../../src/medexrag/observability/) | Prometheus metrics, OTel/Jaeger traces, LangSmith, health |
| Evaluation | [src/medexrag/evaluation/](../../src/medexrag/evaluation/) | E2E benchmarks, retrieval metrics, LLM-judge |

## Two-Stage RAG Workflow

Implemented in `MedicalRAGPipeline.analyze_xray()` ([src/medexrag/pipeline.py](../../src/medexrag/pipeline.py)).

1. VLM runs an initial pass on the X-ray and emits preliminary findings.
2. Findings are used as the query to `MedicalVectorStore.search(k=RAG_K_SOURCES)`.
3. Top-k chunks are concatenated with source metadata.
4. VLM is re-prompted with image + initial findings + retrieved literature.
5. Final report cites sources by filename and page; raw chunks are returned alongside the text for UI display.

## Multi-Agent Layer

Defined in [src/medexrag/agents.py](../../src/medexrag/agents.py) using LangGraph. State flows through four nodes:

- **Analyst** - first-pass image observations.
- **Researcher** - issues literature search tool calls; collects evidence.
- **Diagnostician** - reconciles findings with evidence; ranks differentials.
- **Reporter** - produces the final structured report with citations.

The VLM is exposed to LangChain via a custom LLM wrapper; tools are `literature_search` and `xray_analysis`.

## Defaults

- VLM: `Qwen/Qwen2-VL-2B-Instruct` (override via `MedicalRAGPipeline(model_name=...)`)
- Embeddings: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Chunk size / overlap: 800 / 150 chars
- `RAG_K_SOURCES`: 5
- Vector DB path: `data/medical_kb/` (configurable via `MEDICAL_KB_PATH`)

## Related Docs

- [RAG Guide](../guides/RAG_GUIDE.md) - retrieval and ingestion details
- [VLM Guide](../guides/VLM_GUIDE.md) - model selection and quantization
- [Agents Guide](../guides/AGENTS_GUIDE.md) - LangGraph state machine
- [Observability](../observability/README.md) - metrics, traces, dashboards

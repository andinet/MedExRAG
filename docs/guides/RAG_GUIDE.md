# RAG Guide

Two-stage RAG over medical literature for X-ray interpretation. Pipeline: DocLing PDF parsing -> ChromaDB + PubMedBERT embeddings -> VLM with retrieved context.

Source: [src/medexrag/pipeline.py](../../src/medexrag/pipeline.py)

## Components

| Layer | Class | Location |
|---|---|---|
| PDF parsing | `DocLingProcessor` | [pipeline.py:42](../../src/medexrag/pipeline.py#L42) |
| Vector store | `MedicalVectorStore` | [pipeline.py:133](../../src/medexrag/pipeline.py#L133) |
| VLM | `VLMInference` | [pipeline.py:280](../../src/medexrag/pipeline.py#L280) |
| Orchestrator | `MedicalRAGPipeline` | [pipeline.py:374](../../src/medexrag/pipeline.py#L374) |

Embeddings: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (~400MB, downloaded on first use).
Chunking: 800 chars with 150-char overlap.
Default store: `data/medical_kb/` (ChromaDB, persistent).

## Two-Stage RAG Flow

Implemented in `MedicalRAGPipeline.analyze_xray()` ([pipeline.py:435](../../src/medexrag/pipeline.py#L435)):

1. **Initial analysis** - VLM generates preliminary findings from the X-ray.
2. **Retrieval** - findings are used as the search query against the vector store.
3. **Enhanced analysis** - VLM is re-prompted with image + initial findings + retrieved literature, producing cited output.

Pass `use_rag=False` to skip stages 2-3 (pure VLM).

## CLI

```bash
# Ingest a directory of PDFs
python -m medexrag.cli ingest data/medical_literature/

# RAG-enhanced analysis
python -m medexrag.cli analyze data/xray_images/chest.dcm --rag

# Search the knowledge base
python -m medexrag.cli search "pneumonia consolidation" --k 5

# Batch with RAG
python -m medexrag.cli batch data/xray_images/ --output data/analysis_results/ --rag

# Show vector store stats
python -m medexrag.cli stats
```

CLI source: [src/medexrag/cli.py](../../src/medexrag/cli.py).

## Python API

```python
from medexrag.pipeline import MedicalRAGPipeline

# Ingestion only (no VLM load)
pipeline = MedicalRAGPipeline(load_vlm=False)
pipeline.ingest_directory("data/medical_literature")

# Full pipeline with RAG analysis
pipeline = MedicalRAGPipeline()
result = pipeline.analyze_xray(
    image_path="data/xray_images/chest.dcm",
    question="Evaluate for pneumonia",
    use_rag=True,
    k=5,
)
print(result["enhanced_analysis"])
print(result["sources"])  # citations with relevance scores
```

## Environment Variables

Read from `.env` (created by `setup.py`):

| Variable | Default | Notes |
|---|---|---|
| `MEDICAL_KB_PATH` | `./data/medical_kb` | ChromaDB persist directory |
| `LITERATURE_DIR` | `./data/medical_literature` | PDF source directory |
| `RAG_K_SOURCES` | `5` | Chunks retrieved per query |
| `RAG_SCORE_THRESHOLD` | `0.0` | Min relevance score (0-1); higher = stricter |

Override per call by passing `k=` and `score_threshold=` to `analyze_xray()` or `MedicalVectorStore.search()`.

## Adding Literature

PDFs go in `data/medical_literature/` (gitignored). DocLing OCR is enabled by default so scanned PDFs work. Re-run `ingest` after adding files - chunks are appended to the existing store.

```python
# Single file
pipeline.ingest_pdf("data/medical_literature/acr_guidelines.pdf")

# Directory (recursive)
pipeline.ingest_directory("data/medical_literature")
```

## Troubleshooting

- **Empty search results** - run `examples/test_ingestion.py` to verify the store is populated.
- **Slow first query** - PubMedBERT downloads on first use; cached at `~/.cache/huggingface/`.
- **OCR failures** - check `DocLingProcessor` options at [pipeline.py:42](../../src/medexrag/pipeline.py#L42).

## References

- DocLing: https://github.com/DS4SD/docling
- ChromaDB: https://docs.trychroma.com/
- PubMedBERT: https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

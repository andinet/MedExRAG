# Streamlit App Guide

Interactive web UI for MedExRAG. Source: [src/medexrag/app.py](../../src/medexrag/app.py).

## Run

```bash
pip install -e .
streamlit run src/medexrag/app.py
```

UI: http://localhost:8501. Custom port: `--server.port 8080`.

For Docker deployment, see [DOCKER_SETUP.md](../deployment/DOCKER_SETUP.md).

## Tabs

Tab definitions live near [app.py:590](../../src/medexrag/app.py#L590).

| Tab | Purpose |
|---|---|
| **Chat Analysis** | Conversational X-ray interpretation; toggle RAG on/off, choose number of sources, view citations and confidence per response. |
| **Upload X-Ray** | Drag-and-drop or path-input upload (DICOM `.dcm`, JPEG, PNG); image preview + metadata. |
| **Literature Management** | Upload single PDFs or ingest a directory; semantic search over the knowledge base; live chunk-count stats. |
| **Batch Processing** | Run analysis over a directory of X-rays with progress bar; results exported as JSON to the chosen output directory. |

## Sidebar Settings

- **Knowledge Base Path** - defaults to `data/medical_kb` (ChromaDB).
- **Use RAG Enhancement** - toggle to compare RAG vs. pure VLM output.
- **Sources slider** - number of literature chunks retrieved (1-10, default 5).
- **Clear Chat** - resets `st.session_state.chat_history`.

## Session State Keys

Initialized in `app.py`:

- `chat_history` - list of `{role, content, sources, confidence}` dicts.
- `current_image` - the active `PIL.Image` for chat.
- `pipeline` - cached `MedicalRAGPipeline` instance (see caching note below).
- `analytics` (optional) - per-session counters.

When extending the UI, namespace new keys to avoid collisions with these.

## Caching

The pipeline is loaded once via `@st.cache_resource`. To clear after model or env changes:

- Press `C` in the UI, or
- `streamlit cache clear` in the shell.

First load downloads the VLM (~4 GB) and PubMedBERT (~400 MB) to `~/.cache/huggingface/`; subsequent runs are fast.

## Common Workflow

1. **Literature Management** -> ingest PDFs from `data/medical_literature/` (one-time per dataset).
2. **Upload X-Ray** -> drop a `.dcm`, `.jpg`, or `.png`.
3. **Chat Analysis** -> ask questions; toggle RAG to compare cited vs. uncited answers.
4. **Batch Processing** -> for bulk analysis with JSON export.

## Configuration

Optional `.streamlit/config.toml`:

```toml
[server]
port = 8501
headless = false

[browser]
gatherUsageStats = false
```

Environment variables (`.env`) used by the underlying pipeline: `MEDICAL_KB_PATH`, `LITERATURE_DIR`, `MODEL_NAME`, `RAG_K_SOURCES`. See [RAG_GUIDE.md](RAG_GUIDE.md).

## Troubleshooting

- **App won't start** - `streamlit version` to verify install; Python 3.11+ required.
- **Pipeline stuck loading** - press `C` to clear cache; check VLM is downloading by watching `~/.cache/huggingface/`.
- **Empty search results in Literature tab** - knowledge base is empty; ingest PDFs first.
- **`CUDA out of memory`** - set `quantize=True` in pipeline, or set `CUDA_VISIBLE_DEVICES=""` to force CPU.
- **Image won't upload** - verify format (`.dcm`, `.jpg`, `.png`) and that the file is not corrupted.

## References

- Streamlit docs: https://docs.streamlit.io/
- `@st.cache_resource`: https://docs.streamlit.io/library/api-reference/performance/st.cache_resource

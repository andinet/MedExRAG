# Troubleshooting

Common issues and fixes for MedExRAG. For deeper context see [RAG_ARCHITECTURE](architecture/RAG_ARCHITECTURE.md), [VLM_GUIDE](guides/VLM_GUIDE.md), and [DOCKER_SETUP](deployment/DOCKER_SETUP.md).

## Local Dev

### Search returns no results

Symptom: `vector_store.search(...)` returns empty list even after ingestion.

- Confirm chunks exist: `PYTHONPATH=src python -m medexrag.cli stats`
- Re-run ingestion: `PYTHONPATH=src python examples/test_ingestion.py`
- If using a custom `score_threshold`, lower it; the default is `0.0` (no filter). See `MedicalVectorStore.search()` in [src/medexrag/pipeline.py](../src/medexrag/pipeline.py).

### Import errors / `ModuleNotFoundError: medexrag`

- Set `PYTHONPATH=src` (bash) or `$env:PYTHONPATH="src"` (PowerShell), or
- Install editable: `pip install -e .`

### Streamlit serves stale model or stale pipeline

- Press `C` in the UI to clear `@st.cache_resource`, then refresh
- Or stop and restart `streamlit run src/medexrag/app.py`

### Unicode errors on Windows console

Symptom: `UnicodeEncodeError: 'charmap' codec can't encode ...`

- Use Git Bash / Windows Terminal, or run `chcp 65001` in CMD before launching
- Avoid piping CLI output through legacy `cmd.exe`

## GPU / VRAM

### CUDA out of memory

Symptom: `torch.cuda.OutOfMemoryError` when loading the VLM.

- Default Qwen2-VL-2B at FP16 needs ~4 GB weights + 2-8 GB for visual tokens.
- Force 4-bit quantization (~1.2 GB weights):
  ```python
  MedicalRAGPipeline(load_vlm=True, quantize=True)
  ```
  Requires `pip install bitsandbytes`.
- Quantization is auto-detected on GPUs with <8 GB VRAM (`quantize="auto"`, the default).
- Or fall back to CPU: `CUDA_VISIBLE_DEVICES=""` (slower: ~10-30 s per analysis).
- Or pick a smaller VLM, e.g. `model_name="Qwen/Qwen2-VL-0.5B-Instruct"`.

### VLM inference is very slow

- Confirm CUDA is being used: `python -c "import torch; print(torch.cuda.is_available())"`
- High-resolution X-rays produce many visual tokens. Down-sample large DICOMs before analysis.

## Models

### First-time download stalls or fails

- VLM is ~4 GB, embeddings ~400 MB. Both pull from Hugging Face on first use.
- Check connectivity to `huggingface.co`. Set `HF_HOME` to a writable path if `~/.cache/huggingface/` is restricted.
- Behind a proxy: set `HTTPS_PROXY` and `HF_HUB_ENABLE_HF_TRANSFER=0`.

### `PdfPipelineOptions` AttributeError from DocLing

Older snippets imported `PdfPipelineOptions, InputFormat` from `docling.document_converter`. Use the default converter:
```python
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
```
See `DocLingProcessor` in [src/medexrag/pipeline.py](../src/medexrag/pipeline.py).

## Docker

### Container exits with code 137

Out-of-memory kill. Increase Docker Desktop memory to 12 GB+ (Settings -> Resources).

### Streamlit container won't start

- Inspect logs: `docker compose -f docker/docker-compose.yml logs streamlit`
- Port 8501 already in use: change the host mapping in `docker/docker-compose.yml`.

### GPU not visible inside container

- Use the GPU overlay: `docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up streamlit`
- Verify host has NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Models re-download every run

The `~/.cache/huggingface` directory is mounted as a named volume. If you removed it, models will pull again. Confirm the volume in `docker/docker-compose.yml` is intact.

### Observability services unreachable

- Bring them up explicitly: `docker compose -f docker/docker-compose.yml up prometheus grafana jaeger otel-collector`
- Default URLs: Grafana 3000, Prometheus 9090, Jaeger 16686, app metrics 8002.

## Getting More Help

- Check terminal logs first; most failures print actionable tracebacks.
- Verify dependencies: `pip install -r requirements.txt`.
- Open an issue with the failing command, full traceback, OS, GPU model, and VRAM.

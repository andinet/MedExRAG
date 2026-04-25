# VLM Guide

The VLM layer wraps Qwen2-VL (and compatible vision LLMs) for X-ray inference.

Source: `VLMInference` at [pipeline.py:280](../../src/medexrag/pipeline.py#L280).

## Default Model

`Qwen/Qwen2-VL-2B-Instruct` (~4 GB). Downloaded on first use to `~/.cache/huggingface/`.

## Initialization

```python
from medexrag.pipeline import MedicalRAGPipeline

# Default: VLM + RAG enabled
pipeline = MedicalRAGPipeline()

# Ingestion-only (skip VLM load, saves ~4 GB RAM)
pipeline = MedicalRAGPipeline(load_vlm=False)

# Use a different VLM
pipeline = MedicalRAGPipeline(model_name="Qwen/Qwen2-VL-7B-Instruct")
```

## Quantization

4-bit quantization is auto-detected via `quantize="auto"` (the default). It activates when CUDA is available and free VRAM is below ~8 GB - notably on RTX A1000 (6 GB) and similar.

```python
pipeline = MedicalRAGPipeline(quantize="auto")   # default - decide at load
pipeline = MedicalRAGPipeline(quantize=True)     # force 4-bit
pipeline = MedicalRAGPipeline(quantize=False)    # force FP16
```

Requires `bitsandbytes` (`pip install bitsandbytes`) when active. Detection logic lives in `VLMInference.__init__` at [pipeline.py:280](../../src/medexrag/pipeline.py#L280).

## Device Behavior

`device_map="auto"` is used internally:

- **CUDA available** - model on GPU, FP16 (or 4-bit if quantized). ~2-5 s per inference.
- **CPU only** - model on CPU. ~10-30 s per inference; works but slow.

Force CPU with `CUDA_VISIBLE_DEVICES=""` in the environment.

## Swapping Models

Any vision model with a HF `*ForConditionalGeneration` class and an `AutoProcessor` should work. Tested classes:

- `Qwen2VLForConditionalGeneration`
- `Qwen2_5_VLForConditionalGeneration`

To add a new family, edit the loader branch in `VLMInference.__init__` at [pipeline.py:280](../../src/medexrag/pipeline.py#L280).

## Inference Calls

```python
result = pipeline.analyze_xray(
    image_path="chest.dcm",
    question="Evaluate for pneumonia",
    use_rag=True,
)
# result["initial_analysis"], result["enhanced_analysis"], result["sources"]
```

The VLM is called once when `use_rag=False`, twice when `use_rag=True` (initial + enhanced).

## Configuration via .env

| Variable | Default |
|---|---|
| `MODEL_NAME` | `Qwen/Qwen2-VL-2B-Instruct` |

## Troubleshooting

- **`CUDA out of memory`** - set `quantize=True`, or run on CPU, or pick a smaller model.
- **Slow first call** - model download (~4 GB) on first run. Subsequent runs use the HF cache.
- **`bitsandbytes` import error** - install it (`pip install bitsandbytes`) or set `quantize=False`.
- **Streamlit app stuck on load** - press `C` in the UI to clear `@st.cache_resource`.

## References

- Qwen2-VL: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes

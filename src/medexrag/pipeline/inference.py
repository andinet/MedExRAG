"""VLM inference: local (transformers) and remote (HTTP) variants."""

import time
from typing import Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from medexrag.observability import get_logger
from medexrag.observability.langsmith_config import trace_vlm_call
from medexrag.observability.metrics_config import (
    VLM_INFERENCE_LATENCY,
    VLM_REQUESTS_TOTAL,
    is_metrics_enabled,
    update_model_loaded,
)
from medexrag.observability.tracing import (
    add_span_event,
    trace_operation,
)

logger = get_logger(__name__)

# VRAM threshold (GB) below which 4-bit quantization is auto-enabled.
_QUANTIZE_VRAM_THRESHOLD_GB = 8


def _should_auto_quantize() -> bool:
    """True iff a CUDA GPU is available with less than 8 GB VRAM.

    On CPU we return False because bitsandbytes 4-bit quantization requires CUDA.
    """
    if not torch.cuda.is_available():
        return False
    try:
        vram_bytes = torch.cuda.get_device_properties(0).total_mem
        vram_gb = vram_bytes / (1024**3)
        logger.info(f"GPU VRAM detected: {vram_gb:.1f} GB")
        return vram_gb < _QUANTIZE_VRAM_THRESHOLD_GB
    except Exception as e:
        logger.warning(f"Could not detect VRAM: {e}, defaulting to quantize=True")
        return True


def _resolve_quantize(quantize: Union[bool, str]) -> bool:
    if isinstance(quantize, bool):
        return quantize
    if quantize == "auto":
        result = _should_auto_quantize()
        logger.info(f"Auto-quantize decision: {'enabled' if result else 'disabled'}")
        return result
    raise ValueError(f"quantize must be True, False, or 'auto', got {quantize!r}")


class VLMInference:
    """Local Vision Language Model inference via Hugging Face transformers.

    Defaults to Qwen2-VL; compatible with other Qwen2VL-class models.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "auto",
        quantize: Union[bool, str] = "auto",
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 256 * 28 * 28,
    ):
        logger.info(f"Loading VLM: {model_name}...")

        quantize = _resolve_quantize(quantize)

        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": device,
            "trust_remote_code": True,
        }

        if quantize:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                # device_map must be "auto" for quantized models
                load_kwargs["device_map"] = "auto"
                logger.info("Using 4-bit quantization (bitsandbytes)")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed, falling back to float16. " "Install with: pip install bitsandbytes"
                )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        update_model_loaded(True)
        logger.info("VLM loaded successfully")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 1024,
    ) -> str:
        """Generate a response conditioned on `image` and `prompt`."""
        start_time = time.time()

        with trace_operation(
            "vlm_generate",
            attributes={
                "vlm.prompt_length": len(prompt),
                "vlm.max_new_tokens": max_new_tokens,
                "vlm.image_size": f"{image.size[0]}x{image.size[1]}" if image else "N/A",
                "vlm.model": "qwen2-vl",
            },
        ) as otel_span:
            with trace_vlm_call(
                name="vlm_generate",
                inputs={
                    "prompt_length": len(prompt),
                    "max_new_tokens": max_new_tokens,
                    "image_size": f"{image.size[0]}x{image.size[1]}" if image else "N/A",
                },
                run_type="llm",
                tags=["vlm", "qwen2-vl"],
            ) as run:
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.model.device)

                    add_span_event(
                        "model_inference_start",
                        {"input_tokens": inputs.input_ids.shape[1]},
                    )

                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            repetition_penalty=1.3,
                            no_repeat_ngram_size=5,
                        )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    result = output_text[0]
                    duration_ms = (time.time() - start_time) * 1000
                    duration_seconds = duration_ms / 1000

                    if is_metrics_enabled():
                        if VLM_INFERENCE_LATENCY is not None:
                            VLM_INFERENCE_LATENCY.labels(operation="generate").observe(duration_seconds)
                        if VLM_REQUESTS_TOTAL is not None:
                            VLM_REQUESTS_TOTAL.labels(operation="generate", status="success").inc()

                    otel_span.set_attribute("vlm.output_length", len(result))
                    otel_span.set_attribute("vlm.duration_ms", duration_ms)
                    otel_span.set_attribute("vlm.input_tokens", inputs.input_ids.shape[1])

                    logger.info(
                        "vlm_generate_complete",
                        duration_ms=duration_ms,
                        output_length=len(result),
                        input_tokens=inputs.input_ids.shape[1],
                    )

                    run.end(
                        outputs={
                            "response_length": len(result),
                            "duration_ms": duration_ms,
                            "input_tokens": inputs.input_ids.shape[1],
                        }
                    )

                    return result

                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    if is_metrics_enabled() and VLM_REQUESTS_TOTAL is not None:
                        VLM_REQUESTS_TOTAL.labels(operation="generate", status="error").inc()
                    logger.error("vlm_generate_failed", error=str(e), duration_ms=duration_ms)
                    run.end(error=str(e))
                    raise


class RemoteVLMInference:
    """HTTP client for a separately-deployed VLM server.

    Drop-in replacement for VLMInference; used in Kubernetes deployments where
    the VLM runs on a dedicated GPU pod.
    """

    def __init__(self, endpoint: str = "http://vlm-worker:8080"):
        import requests as _requests

        self._requests = _requests
        self.endpoint = endpoint.rstrip("/")
        logger.info(f"Using remote VLM at: {self.endpoint}")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 1024,
    ) -> str:
        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        response = self._requests.post(
            f"{self.endpoint}/analyze",
            json={
                "image_b64": image_b64,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            },
            # 5-min timeout to allow for cold start of GPU node
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"]

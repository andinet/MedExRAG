"""
FastAPI VLM Inference Server

Standalone HTTP service wrapping VLMInference for remote inference.
Used in Kubernetes deployments where the VLM runs on a GPU node
and the Streamlit app runs on a separate CPU node.

Usage:
    uvicorn medexrag.vlm_server:app --host 0.0.0.0 --port 8080

Endpoints:
    POST /analyze  — Accept base64 image + prompt, return VLM response
    GET  /health   — Basic health check (is the server up?)
    GET  /ready    — Readiness check (is the model loaded and warm?)
"""

import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global VLM instance (loaded on startup)
_vlm = None
_model_name = None


class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint."""

    image_b64: str = Field(..., description="Base64-encoded image (PNG/JPEG)")
    prompt: str = Field(..., description="Text prompt for VLM")
    max_new_tokens: int = Field(default=1024, ge=1, le=4096)


class AnalyzeResponse(BaseModel):
    """Response body for /analyze endpoint."""

    response: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    model_loaded: bool
    model_name: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load VLM model on startup."""
    global _vlm, _model_name
    import os

    from medexrag.pipeline import VLMInference

    _model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
    logger.info(f"Loading VLM model: {_model_name}")
    _vlm = VLMInference(model_name=_model_name)
    logger.info("VLM model loaded successfully")
    yield
    logger.info("Shutting down VLM server")


app = FastAPI(
    title="MedExRAG VLM Server",
    description="Vision Language Model inference service for medical X-ray analysis",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Analyze an image with the VLM.

    Accepts a base64-encoded image and a text prompt.
    Returns the VLM-generated response.
    """
    if _vlm is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        image_bytes = base64.b64decode(request.image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    start = time.time()
    result = _vlm.generate(
        image=image,
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )
    elapsed_ms = (time.time() - start) * 1000

    return AnalyzeResponse(response=result, inference_time_ms=elapsed_ms)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Basic health check — returns 200 if the server is running."""
    return HealthResponse(
        status="healthy",
        model_loaded=_vlm is not None,
        model_name=_model_name,
    )


@app.get("/ready", response_model=HealthResponse)
async def ready():
    """Readiness check — returns 200 only when the model is fully loaded.

    Used as Kubernetes readiness probe so traffic is only routed
    to this pod after the model is warm.
    """
    if _vlm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready — still loading",
        )
    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_name=_model_name,
    )

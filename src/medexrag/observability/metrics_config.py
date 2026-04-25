"""
Prometheus Metrics Configuration for Medical X-Ray RAG System

Provides metrics collection for:
- VLM inference latency and request counts
- Vector store search latency
- Pipeline total latency and success/error rates
- Active request tracking

Usage:
    from observability.metrics_config import setup_metrics, start_metrics_server

    # Initialize metrics
    setup_metrics()

    # Start metrics HTTP server (for Prometheus scraping)
    start_metrics_server(port=8000)

    # Use metrics in code
    from observability.metrics_config import (
        VLM_INFERENCE_LATENCY,
        ANALYSIS_REQUESTS_TOTAL,
        record_vlm_inference,
        record_pipeline_request
    )

    # Record VLM inference
    with record_vlm_inference("initial_analysis"):
        result = vlm.generate(...)

    # Or manually observe histogram
    VLM_INFERENCE_LATENCY.labels(operation="initial_analysis").observe(duration)
"""

import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable

# Prometheus imports with graceful degradation
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for when prometheus_client is not installed
    Counter = None
    Histogram = None
    Gauge = None
    Info = None

# Module-level state
_metrics_initialized = False
_metrics_server_started = False
_metrics_lock = threading.Lock()

# =============================================================================
# Metric Definitions
# =============================================================================

if PROMETHEUS_AVAILABLE:
    # -------------------------------------------------------------------------
    # Counters - Track cumulative events
    # -------------------------------------------------------------------------
    ANALYSIS_REQUESTS_TOTAL = Counter(
        "medical_rag_analysis_requests_total",
        "Total X-ray analysis requests",
        ["use_rag", "status"],  # labels: use_rag=true/false, status=success/error
    )

    LITERATURE_SEARCHES_TOTAL = Counter(
        "medical_rag_literature_searches_total",
        "Total literature search operations",
        ["has_threshold"],  # has_threshold=true/false
    )

    VLM_REQUESTS_TOTAL = Counter(
        "medical_rag_vlm_requests_total",
        "Total VLM inference requests",
        ["operation", "status"],  # operation=initial_analysis/enhanced_analysis, status=success/error
    )

    LITERATURE_INGESTION_TOTAL = Counter(
        "medical_rag_literature_ingestion_total",
        "Total literature documents ingested",
        ["status"],  # status=success/error
    )

    # -------------------------------------------------------------------------
    # Histograms - Track latency distributions
    # -------------------------------------------------------------------------
    VLM_INFERENCE_LATENCY = Histogram(
        "medical_rag_vlm_inference_seconds",
        "VLM inference latency in seconds",
        ["operation"],  # operation=initial_analysis/enhanced_analysis/generate
        buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    )

    VECTOR_SEARCH_LATENCY = Histogram(
        "medical_rag_vector_search_seconds",
        "Vector store search latency in seconds",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )

    PIPELINE_TOTAL_LATENCY = Histogram(
        "medical_rag_pipeline_total_seconds",
        "Total pipeline execution time in seconds",
        ["use_rag"],  # use_rag=true/false
        buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    )

    LITERATURE_RETRIEVAL_LATENCY = Histogram(
        "medical_rag_literature_retrieval_seconds",
        "Literature retrieval step latency in seconds",
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    AGENT_STEP_DURATION = Histogram(
        "medical_rag_agent_step_seconds",
        "Duration of each agent step in seconds",
        ["agent_name"],  # agent_name=analyst/researcher/diagnostician/reporter
        buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    )

    # -------------------------------------------------------------------------
    # Counters - Additional (M2 agents)
    # -------------------------------------------------------------------------
    TOOL_INVOCATIONS = Counter(
        "medical_rag_tool_invocations_total",
        "Total tool invocations by agents",
        [
            "tool_name",
            "status",
        ],  # tool_name=image_analysis/literature_search/diagnostic_reasoning, status=success/error
    )

    # -------------------------------------------------------------------------
    # Gauges - Track current state
    # -------------------------------------------------------------------------
    VECTOR_STORE_CHUNKS = Gauge("medical_rag_vector_store_chunks_total", "Total chunks in vector store")

    MODEL_LOADED = Gauge("medical_rag_model_loaded", "Whether the VLM model is loaded (1=yes, 0=no)")

    ACTIVE_REQUESTS = Gauge("medical_rag_active_requests", "Currently processing analysis requests")

    SOURCES_RETRIEVED = Gauge("medical_rag_sources_retrieved_last", "Number of sources retrieved in last search")

    # -------------------------------------------------------------------------
    # Info - Service metadata
    # -------------------------------------------------------------------------
    SERVICE_INFO = Info("medical_rag_service", "Medical RAG service information")

else:
    # Dummy metrics when prometheus_client not available
    ANALYSIS_REQUESTS_TOTAL = None
    LITERATURE_SEARCHES_TOTAL = None
    VLM_REQUESTS_TOTAL = None
    LITERATURE_INGESTION_TOTAL = None
    VLM_INFERENCE_LATENCY = None
    VECTOR_SEARCH_LATENCY = None
    PIPELINE_TOTAL_LATENCY = None
    LITERATURE_RETRIEVAL_LATENCY = None
    AGENT_STEP_DURATION = None
    TOOL_INVOCATIONS = None
    VECTOR_STORE_CHUNKS = None
    MODEL_LOADED = None
    ACTIVE_REQUESTS = None
    SOURCES_RETRIEVED = None
    SERVICE_INFO = None


# =============================================================================
# Setup Functions
# =============================================================================


def setup_metrics(service_name: str = "medical-rag") -> bool:
    """
    Initialize Prometheus metrics.

    Args:
        service_name: Name of the service for metadata

    Returns:
        True if metrics were initialized successfully
    """
    global _metrics_initialized

    if not PROMETHEUS_AVAILABLE:
        print("Warning: prometheus_client not installed. Metrics disabled.")
        return False

    if _metrics_initialized:
        return True

    with _metrics_lock:
        if _metrics_initialized:
            return True

        try:
            # Set service info
            SERVICE_INFO.info(
                {
                    "service_name": service_name,
                    "version": "1.0.0",
                    "model": os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct"),
                }
            )

            _metrics_initialized = True
            print(f"Prometheus metrics initialized for service: {service_name}")
            return True

        except Exception as e:
            print(f"Warning: Failed to initialize Prometheus metrics: {e}")
            return False


def start_metrics_server(port: int = None) -> bool:
    """
    Start the Prometheus metrics HTTP server.

    Args:
        port: Port to expose metrics on (default: METRICS_PORT env var or 8000)

    Returns:
        True if server started successfully
    """
    global _metrics_server_started

    if not PROMETHEUS_AVAILABLE:
        return False

    if _metrics_server_started:
        return True

    with _metrics_lock:
        if _metrics_server_started:
            return True

        port = port or int(os.getenv("METRICS_PORT", "8000"))

        try:
            start_http_server(port)
            _metrics_server_started = True
            print(f"Prometheus metrics server started on port {port}")
            return True

        except Exception as e:
            print(f"Warning: Failed to start metrics server on port {port}: {e}")
            return False


def is_metrics_enabled() -> bool:
    """Check if Prometheus metrics are available and initialized."""
    return PROMETHEUS_AVAILABLE and _metrics_initialized


# =============================================================================
# Context Managers and Decorators for Instrumentation
# =============================================================================


@contextmanager
def record_vlm_inference(operation: str = "generate"):
    """
    Context manager to record VLM inference latency and counts.

    Args:
        operation: Type of operation (initial_analysis, enhanced_analysis, generate)

    Example:
        with record_vlm_inference("initial_analysis"):
            result = vlm.generate(image, prompt)
    """
    if not is_metrics_enabled():
        yield
        return

    start_time = time.time()
    status = "success"

    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.time() - start_time
        VLM_INFERENCE_LATENCY.labels(operation=operation).observe(duration)
        VLM_REQUESTS_TOTAL.labels(operation=operation, status=status).inc()


@contextmanager
def record_vector_search():
    """
    Context manager to record vector search latency.

    Example:
        with record_vector_search():
            results = vector_store.search(query)
    """
    if not is_metrics_enabled():
        yield
        return

    start_time = time.time()

    try:
        yield
    finally:
        duration = time.time() - start_time
        VECTOR_SEARCH_LATENCY.observe(duration)


@contextmanager
def record_pipeline_request(use_rag: bool = False):
    """
    Context manager to record full pipeline execution.

    Args:
        use_rag: Whether RAG is enabled for this request

    Example:
        with record_pipeline_request(use_rag=True):
            result = pipeline.analyze_xray(image, question)
    """
    if not is_metrics_enabled():
        yield
        return

    start_time = time.time()
    status = "success"
    rag_label = "true" if use_rag else "false"

    # Increment active requests
    ACTIVE_REQUESTS.inc()

    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.time() - start_time

        # Record metrics
        PIPELINE_TOTAL_LATENCY.labels(use_rag=rag_label).observe(duration)
        ANALYSIS_REQUESTS_TOTAL.labels(use_rag=rag_label, status=status).inc()

        # Decrement active requests
        ACTIVE_REQUESTS.dec()


@contextmanager
def record_literature_retrieval():
    """
    Context manager to record literature retrieval latency.

    Example:
        with record_literature_retrieval():
            sources = pipeline.search_literature(query)
    """
    if not is_metrics_enabled():
        yield
        return

    start_time = time.time()

    try:
        yield
    finally:
        duration = time.time() - start_time
        LITERATURE_RETRIEVAL_LATENCY.observe(duration)


def track_vlm_inference(operation: str = "generate") -> Callable:
    """
    Decorator to track VLM inference metrics.

    Args:
        operation: Type of operation for labeling

    Example:
        @track_vlm_inference("initial_analysis")
        def analyze_image(image, prompt):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with record_vlm_inference(operation):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def track_pipeline_request(use_rag_arg: str = None) -> Callable:
    """
    Decorator to track pipeline request metrics.

    Args:
        use_rag_arg: Name of the argument that indicates RAG usage (default: 'use_rag')

    Example:
        @track_pipeline_request(use_rag_arg='use_rag')
        def analyze_xray(self, image, question, use_rag=False):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine if RAG is enabled
            use_rag = kwargs.get(use_rag_arg or "use_rag", False)
            with record_pipeline_request(use_rag=use_rag):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Helper Functions for Manual Metric Updates
# =============================================================================


def update_vector_store_chunks(count: int) -> None:
    """Update the vector store chunks gauge."""
    if is_metrics_enabled():
        VECTOR_STORE_CHUNKS.set(count)


def update_model_loaded(loaded: bool) -> None:
    """Update the model loaded gauge."""
    if is_metrics_enabled():
        MODEL_LOADED.set(1 if loaded else 0)


def update_sources_retrieved(count: int) -> None:
    """Update the sources retrieved gauge."""
    if is_metrics_enabled():
        SOURCES_RETRIEVED.set(count)


def record_literature_search(has_threshold: bool = False) -> None:
    """Increment the literature search counter."""
    if is_metrics_enabled():
        LITERATURE_SEARCHES_TOTAL.labels(has_threshold="true" if has_threshold else "false").inc()


def record_ingestion(status: str = "success") -> None:
    """Increment the literature ingestion counter."""
    if is_metrics_enabled():
        LITERATURE_INGESTION_TOTAL.labels(status=status).inc()


# =============================================================================
# Agent Metrics (M2 - Multi-Agent Workflow)
# =============================================================================


@contextmanager
def record_agent_step(agent_name: str):
    """
    Context manager to record agent step duration.

    Args:
        agent_name: Name of the agent (analyst, researcher, diagnostician, reporter)

    Example:
        with record_agent_step("analyst"):
            result = analyst_agent.run(state)
    """
    if not is_metrics_enabled():
        yield
        return

    start_time = time.time()

    try:
        yield
    finally:
        duration = time.time() - start_time
        if AGENT_STEP_DURATION is not None:
            AGENT_STEP_DURATION.labels(agent_name=agent_name).observe(duration)


def record_tool_invocation(tool_name: str, status: str = "success") -> None:
    """
    Record a tool invocation.

    Args:
        tool_name: Name of the tool (image_analysis, literature_search, diagnostic_reasoning)
        status: 'success' or 'error'

    Example:
        record_tool_invocation("literature_search", "success")
    """
    if is_metrics_enabled() and TOOL_INVOCATIONS is not None:
        TOOL_INVOCATIONS.labels(tool_name=tool_name, status=status).inc()


def track_agent_step(agent_name: str) -> Callable:
    """
    Decorator to track agent step metrics.

    Args:
        agent_name: Name of the agent for labeling

    Example:
        @track_agent_step("analyst")
        def analyst_agent(self, state):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with record_agent_step(agent_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator

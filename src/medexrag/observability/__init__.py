"""
Observability package for Medical X-Ray RAG Analysis

This package provides comprehensive observability features:
- Structured logging with structlog (JSON format, correlation IDs)
- LangSmith integration for LLM/agent tracing
- Prometheus metrics for performance monitoring (M2)
- OpenTelemetry distributed tracing (M3)

Usage:
    from observability import setup_observability, get_logger

    # Initialize all observability features
    setup_observability()

    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("analysis_started", image_path="/path/to/image.dcm")

    # For Prometheus metrics
    from observability import start_metrics_server, record_vlm_inference

    # Start metrics HTTP server (typically in main app)
    start_metrics_server(port=8000)

    # Use context managers for instrumentation
    with record_vlm_inference("initial_analysis"):
        result = vlm.generate(...)
"""

import logging
import os

# Module-level flag to track initialization
_initialized = False


def setup_observability(
    service_name: str = "medical-rag",
    log_level: str = None,
    log_format: str = None,
    enable_langsmith: bool = None,
    enable_metrics: bool = None,
) -> bool:
    """
    Initialize all observability components with graceful degradation.

    Components are enabled based on environment variables:
    - LANGSMITH_ENABLED / LANGCHAIN_TRACING_V2: Enable LangSmith tracing
    - PROMETHEUS_ENABLED: Enable Prometheus metrics (M2)
    - LOG_LEVEL: Set logging level (default: INFO)
    - LOG_FORMAT: Set log format - 'json' or 'console' (default: json)

    Args:
        service_name: Name of the service for tracing
        log_level: Override LOG_LEVEL env var
        log_format: Override LOG_FORMAT env var
        enable_langsmith: Override LANGSMITH_ENABLED env var
        enable_metrics: Override PROMETHEUS_ENABLED env var

    Returns:
        True if all requested components initialized successfully
    """
    global _initialized

    if _initialized:
        return True

    errors = []

    # Determine settings from args or environment
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    log_format = log_format or os.getenv("LOG_FORMAT", "json")

    # Setup structured logging (always enabled)
    try:
        from .logging_config import setup_logging

        setup_logging(log_level=log_level, log_format=log_format)
    except Exception as e:
        errors.append(f"Logging setup failed: {e}")
        # Fall back to basic logging
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

    # Setup LangSmith (optional)
    langsmith_enabled = enable_langsmith
    if langsmith_enabled is None:
        langsmith_enabled = (
            os.getenv("LANGSMITH_ENABLED", "").lower() == "true"
            or os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        )

    if langsmith_enabled:
        try:
            from .langsmith_config import setup_langsmith

            setup_langsmith()
        except Exception as e:
            # Log warning but don't fail - app should work without tracing
            logging.warning(f"LangSmith setup failed, continuing without tracing: {e}")

    # Setup Prometheus metrics (optional - M2)
    metrics_enabled = enable_metrics
    if metrics_enabled is None:
        metrics_enabled = os.getenv("PROMETHEUS_ENABLED", "").lower() == "true"

    if metrics_enabled:
        try:
            from .metrics_config import setup_metrics

            setup_metrics(service_name=service_name)
        except Exception as e:
            # Log warning but don't fail - app should work without metrics
            logging.warning(f"Prometheus metrics setup failed, continuing without metrics: {e}")

    # Setup OpenTelemetry distributed tracing (optional - M3)
    otel_enabled = os.getenv("OTEL_ENABLED", "").lower() == "true"

    if otel_enabled:
        try:
            from .tracing import setup_tracing

            setup_tracing(service_name=service_name)
        except Exception as e:
            # Log warning but don't fail - app should work without tracing
            logging.warning(f"OpenTelemetry tracing setup failed, continuing without distributed tracing: {e}")

    _initialized = True

    if errors:
        logging.error(f"Observability setup had errors: {errors}")
        return False

    return True


def get_logger(name: str):
    """
    Get a structured logger for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        A structlog logger instance
    """
    try:
        from .logging_config import get_logger as _get_logger

        return _get_logger(name)
    except ImportError:
        # Fallback to standard logging if structlog not configured
        return logging.getLogger(name)


# Re-export commonly used functions
try:
    from .langsmith_config import create_langsmith_callback, get_langsmith_client, setup_langsmith, trace_vlm_call
    from .logging_config import get_logger, set_correlation_id, setup_logging
except ImportError:
    # Allow partial imports during development
    pass

# Re-export OpenTelemetry tracing functions (M3)
try:
    from .context import (
        clear_request_context,
        generate_request_id,
        generate_session_id,
        get_context_dict,
        get_request_id,
        get_session_id,
        inject_context_to_trace,
        request_context,
        set_request_context,
        set_request_id,
        set_session_id,
    )
    from .tracing import (
        NoOpSpan,
        add_span_attribute,
        add_span_event,
        get_current_span,
        get_current_span_id,
        get_current_trace_id,
        get_tracer,
        is_tracing_enabled,
        setup_tracing,
        trace_function,
        trace_operation,
    )
except ImportError:
    # Allow partial imports during development
    pass

# Re-export Prometheus metrics functions (M2)
try:
    from .metrics_config import (  # Context managers for instrumentation; Decorators; Helper functions; Metric instances (for direct access if needed)
        ACTIVE_REQUESTS,
        AGENT_STEP_DURATION,
        ANALYSIS_REQUESTS_TOTAL,
        MODEL_LOADED,
        PIPELINE_TOTAL_LATENCY,
        TOOL_INVOCATIONS,
        VECTOR_SEARCH_LATENCY,
        VECTOR_STORE_CHUNKS,
        VLM_INFERENCE_LATENCY,
        VLM_REQUESTS_TOTAL,
        is_metrics_enabled,
        record_agent_step,
        record_ingestion,
        record_literature_retrieval,
        record_literature_search,
        record_pipeline_request,
        record_tool_invocation,
        record_vector_search,
        record_vlm_inference,
        setup_metrics,
        start_metrics_server,
        track_agent_step,
        track_pipeline_request,
        track_vlm_inference,
        update_model_loaded,
        update_sources_retrieved,
        update_vector_store_chunks,
    )
except ImportError:
    # Allow partial imports during development
    pass

# Re-export Health Check functions (M4)
try:
    from .health import (
        ComponentHealth,
        HealthChecker,
        HealthStatus,
        SystemHealth,
        configure_health_checker,
        get_health_checker,
    )
except ImportError:
    # Allow partial imports during development
    pass


__all__ = [
    # Core setup
    "setup_observability",
    # Logging (M1)
    "setup_logging",
    "get_logger",
    "set_correlation_id",
    # LangSmith (M1)
    "setup_langsmith",
    "create_langsmith_callback",
    "trace_vlm_call",
    "get_langsmith_client",
    # Prometheus Metrics (M2)
    "setup_metrics",
    "start_metrics_server",
    "is_metrics_enabled",
    "record_vlm_inference",
    "record_vector_search",
    "record_pipeline_request",
    "record_literature_retrieval",
    "record_agent_step",
    "track_vlm_inference",
    "track_pipeline_request",
    "track_agent_step",
    "update_vector_store_chunks",
    "update_model_loaded",
    "update_sources_retrieved",
    "record_literature_search",
    "record_ingestion",
    "record_tool_invocation",
    "ANALYSIS_REQUESTS_TOTAL",
    "VLM_INFERENCE_LATENCY",
    "VLM_REQUESTS_TOTAL",
    "VECTOR_SEARCH_LATENCY",
    "PIPELINE_TOTAL_LATENCY",
    "ACTIVE_REQUESTS",
    "VECTOR_STORE_CHUNKS",
    "MODEL_LOADED",
    "AGENT_STEP_DURATION",
    "TOOL_INVOCATIONS",
    # OpenTelemetry Distributed Tracing (M3)
    "setup_tracing",
    "is_tracing_enabled",
    "get_tracer",
    "trace_operation",
    "trace_function",
    "get_current_span",
    "get_current_trace_id",
    "get_current_span_id",
    "add_span_attribute",
    "add_span_event",
    "NoOpSpan",
    # Request Context (M3)
    "generate_request_id",
    "generate_session_id",
    "set_request_id",
    "get_request_id",
    "set_session_id",
    "get_session_id",
    "set_request_context",
    "clear_request_context",
    "get_context_dict",
    "request_context",
    "inject_context_to_trace",
    # Health Checks (M4)
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "get_health_checker",
    "configure_health_checker",
]

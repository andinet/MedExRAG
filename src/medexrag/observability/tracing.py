"""
OpenTelemetry Distributed Tracing for Medical X-Ray RAG Application

This module provides distributed tracing capabilities using OpenTelemetry,
enabling request correlation and latency analysis across the entire pipeline.

Usage:
    from observability.tracing import setup_tracing, trace_operation, get_tracer

    # Initialize tracing (typically in main app)
    setup_tracing(service_name="medical-rag")

    # Use context manager for tracing
    with trace_operation("vlm_inference", {"model": "qwen2-vl"}) as span:
        result = model.generate(...)
        span.set_attribute("tokens_generated", len(result))

    # Or use decorator
    @trace_function("process_image")
    def process_image(image_path: str):
        ...
"""

import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Span, Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Span = None

logger = logging.getLogger(__name__)

# Module-level state
_tracer: Optional[Any] = None
_initialized: bool = False


def is_tracing_enabled() -> bool:
    """Check if OpenTelemetry tracing is enabled and initialized."""
    return _initialized and OTEL_AVAILABLE


def setup_tracing(
    service_name: str = "medical-rag",
    otlp_endpoint: str = None,
    enable_console_export: bool = False,
) -> bool:
    """
    Initialize OpenTelemetry tracing with OTLP exporter.

    Args:
        service_name: Name of the service for trace identification
        otlp_endpoint: OTLP collector endpoint (default from env or localhost:4317)
        enable_console_export: Also print spans to console (for debugging)

    Returns:
        True if tracing was successfully initialized
    """
    global _tracer, _initialized

    if _initialized:
        return True

    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry packages not installed. Tracing disabled.")
        return False

    # Check if tracing is enabled via environment
    otel_enabled = os.getenv("OTEL_ENABLED", "").lower() == "true"
    if not otel_enabled:
        logger.info("OpenTelemetry tracing disabled (OTEL_ENABLED != true)")
        return False

    try:
        # Get OTLP endpoint from env or parameter
        endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

        # Create resource with service information
        resource = Resource.create(
            {
                SERVICE_NAME: service_name,
                "service.version": "1.0.0",
                "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            }
        )

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add OTLP exporter (sends to collector)
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=endpoint,
                insecure=True,  # For local development; use TLS in production
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP exporter configured for endpoint: {endpoint}")
        except Exception as e:
            logger.warning(f"Failed to configure OTLP exporter: {e}")

        # Optionally add console exporter for debugging
        if enable_console_export or os.getenv("OTEL_CONSOLE_EXPORT", "").lower() == "true":
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))
            logger.info("Console span exporter enabled")

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer for this module
        _tracer = trace.get_tracer(__name__, "1.0.0")

        _initialized = True
        logger.info(f"OpenTelemetry tracing initialized for service: {service_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
        return False


def get_tracer(name: str = __name__) -> Optional[Any]:
    """
    Get a tracer instance for creating spans.

    Args:
        name: Name for the tracer (typically __name__ of the module)

    Returns:
        OpenTelemetry tracer or None if not initialized
    """
    if not is_tracing_enabled():
        return None

    return trace.get_tracer(name, "1.0.0")


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: Dict[str, Any] = None,
    record_exception: bool = True,
):
    """
    Context manager for tracing an operation with automatic error recording.

    Args:
        operation_name: Name of the operation (becomes span name)
        attributes: Optional dict of attributes to attach to span
        record_exception: Whether to record exceptions as span events

    Yields:
        The span object (or a no-op object if tracing disabled)

    Example:
        with trace_operation("vlm_inference", {"model": "qwen2-vl"}) as span:
            result = model.generate(prompt)
            span.set_attribute("output_length", len(result))
    """
    if not is_tracing_enabled():
        # Yield a no-op span-like object
        yield NoOpSpan()
        return

    tracer = get_tracer()
    if tracer is None:
        yield NoOpSpan()
        return

    with tracer.start_as_current_span(operation_name) as span:
        try:
            # Set initial attributes
            if attributes:
                for key, value in attributes.items():
                    if value is not None:
                        span.set_attribute(key, _sanitize_attribute(value))

            yield span

            # Mark as successful
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            # Record the exception
            if record_exception:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def trace_function(
    operation_name: str = None,
    attributes: Dict[str, Any] = None,
):
    """
    Decorator for tracing a function.

    Args:
        operation_name: Name for the span (defaults to function name)
        attributes: Static attributes to add to every span

    Example:
        @trace_function("process_xray")
        def analyze_xray(image_path: str):
            ...
    """

    def decorator(func: Callable) -> Callable:
        span_name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_attributes = dict(attributes) if attributes else {}
            span_attributes["function.name"] = func.__name__

            with trace_operation(span_name, span_attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_current_span() -> Optional[Any]:
    """Get the current active span, if any."""
    if not is_tracing_enabled():
        return None
    return trace.get_current_span()


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID as a hex string."""
    if not is_tracing_enabled():
        return None

    span = trace.get_current_span()
    if span is None:
        return None

    span_context = span.get_span_context()
    if span_context.is_valid:
        return format(span_context.trace_id, "032x")
    return None


def get_current_span_id() -> Optional[str]:
    """Get the current span ID as a hex string."""
    if not is_tracing_enabled():
        return None

    span = trace.get_current_span()
    if span is None:
        return None

    span_context = span.get_span_context()
    if span_context.is_valid:
        return format(span_context.span_id, "016x")
    return None


def add_span_attribute(key: str, value: Any) -> None:
    """Add an attribute to the current span."""
    if not is_tracing_enabled():
        return

    span = trace.get_current_span()
    if span:
        span.set_attribute(key, _sanitize_attribute(value))


def add_span_event(name: str, attributes: Dict[str, Any] = None) -> None:
    """Add an event to the current span."""
    if not is_tracing_enabled():
        return

    span = trace.get_current_span()
    if span:
        sanitized = {}
        if attributes:
            for k, v in attributes.items():
                sanitized[k] = _sanitize_attribute(v)
        span.add_event(name, sanitized)


def _sanitize_attribute(value: Any) -> Any:
    """
    Sanitize attribute values for OpenTelemetry.
    OTEL only accepts str, bool, int, float, or sequences of these.
    """
    if value is None:
        return "null"
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_sanitize_attribute(v) for v in value]
    # Convert other types to string
    return str(value)


class NoOpSpan:
    """
    A no-op span implementation for when tracing is disabled.
    Allows code to use span methods without checking if tracing is enabled.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: Dict[str, Any] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def get_span_context(self) -> None:
        return None

    def is_recording(self) -> bool:
        return False


# Convenience aliases
start_span = trace_operation
instrument = trace_function


__all__ = [
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
    "start_span",
    "instrument",
]

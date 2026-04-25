"""
Request Context Management for Medical X-Ray RAG Application

This module provides request-scoped context using Python's contextvars,
enabling correlation of logs, metrics, and traces across the pipeline.

Usage:
    from observability.context import (
        set_request_context,
        get_request_id,
        get_session_id,
        request_context,
    )

    # Set context for a request
    with request_context(request_id="req-123", session_id="sess-456"):
        # All operations within this block share the same context
        logger.info("Processing request")  # Will include request_id
        analyze_xray(...)

    # Or manually
    set_request_context(request_id="req-123")
    process()
    clear_request_context()
"""

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Context variables for request tracking
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_request_start_time: ContextVar[Optional[datetime]] = ContextVar("request_start_time", default=None)
_request_attributes: ContextVar[Dict[str, Any]] = ContextVar("request_attributes", default={})


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req-{uuid.uuid4().hex[:12]}"


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"sess-{uuid.uuid4().hex[:16]}"


def set_request_id(request_id: str) -> None:
    """Set the current request ID."""
    _request_id.set(request_id)


def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    return _request_id.get()


def set_session_id(session_id: str) -> None:
    """Set the current session ID."""
    _session_id.set(session_id)


def get_session_id() -> Optional[str]:
    """Get the current session ID."""
    return _session_id.get()


def set_user_id(user_id: str) -> None:
    """Set the current user ID."""
    _user_id.set(user_id)


def get_user_id() -> Optional[str]:
    """Get the current user ID."""
    return _user_id.get()


def set_request_start_time(start_time: datetime = None) -> None:
    """Set the request start time."""
    _request_start_time.set(start_time or datetime.utcnow())


def get_request_start_time() -> Optional[datetime]:
    """Get the request start time."""
    return _request_start_time.get()


def get_request_duration_ms() -> Optional[float]:
    """Get the duration since request start in milliseconds."""
    start = _request_start_time.get()
    if start is None:
        return None
    delta = datetime.utcnow() - start
    return delta.total_seconds() * 1000


def set_request_attribute(key: str, value: Any) -> None:
    """Set a custom attribute on the current request context."""
    attrs = _request_attributes.get()
    if attrs is None:
        attrs = {}
    attrs[key] = value
    _request_attributes.set(attrs)


def get_request_attribute(key: str, default: Any = None) -> Any:
    """Get a custom attribute from the current request context."""
    attrs = _request_attributes.get()
    if attrs is None:
        return default
    return attrs.get(key, default)


def get_all_request_attributes() -> Dict[str, Any]:
    """Get all custom attributes from the current request context."""
    return dict(_request_attributes.get() or {})


def set_request_context(
    request_id: str = None,
    session_id: str = None,
    user_id: str = None,
) -> str:
    """
    Set the full request context. Generates request_id if not provided.

    Args:
        request_id: Unique request identifier (generated if None)
        session_id: Session identifier for grouping requests
        user_id: User identifier

    Returns:
        The request_id (generated or provided)
    """
    req_id = request_id or generate_request_id()
    _request_id.set(req_id)
    _request_start_time.set(datetime.utcnow())
    _request_attributes.set({})

    if session_id:
        _session_id.set(session_id)
    if user_id:
        _user_id.set(user_id)

    return req_id


def clear_request_context() -> None:
    """Clear all request context variables."""
    _request_id.set(None)
    _session_id.set(None)
    _user_id.set(None)
    _request_start_time.set(None)
    _request_attributes.set({})


def get_context_dict() -> Dict[str, Any]:
    """
    Get all context values as a dictionary.
    Useful for injecting into logs or trace attributes.
    """
    context = {}

    request_id = _request_id.get()
    if request_id:
        context["request_id"] = request_id

    session_id = _session_id.get()
    if session_id:
        context["session_id"] = session_id

    user_id = _user_id.get()
    if user_id:
        context["user_id"] = user_id

    start_time = _request_start_time.get()
    if start_time:
        context["request_start_time"] = start_time.isoformat()

    # Include custom attributes
    attrs = _request_attributes.get()
    if attrs:
        context.update(attrs)

    return context


@contextmanager
def request_context(
    request_id: str = None,
    session_id: str = None,
    user_id: str = None,
):
    """
    Context manager for setting request context.
    Automatically clears context when exiting.

    Args:
        request_id: Unique request identifier (generated if None)
        session_id: Session identifier
        user_id: User identifier

    Yields:
        The request_id

    Example:
        with request_context(session_id="user-session-123") as req_id:
            logger.info(f"Processing request {req_id}")
            result = analyze_xray(image_path)
    """
    req_id = set_request_context(
        request_id=request_id,
        session_id=session_id,
        user_id=user_id,
    )
    try:
        yield req_id
    finally:
        clear_request_context()


def inject_context_to_trace(span) -> None:
    """
    Inject current context values as span attributes.

    Args:
        span: OpenTelemetry span to add attributes to
    """
    if span is None:
        return

    context = get_context_dict()
    for key, value in context.items():
        try:
            if value is not None:
                span.set_attribute(f"context.{key}", str(value))
        except Exception:
            pass  # Ignore attribute setting errors


def inject_context_to_log(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inject current context values into a log record.

    Args:
        record: Log record dictionary

    Returns:
        Updated log record with context values
    """
    context = get_context_dict()
    record.update(context)
    return record


__all__ = [
    "generate_request_id",
    "generate_session_id",
    "set_request_id",
    "get_request_id",
    "set_session_id",
    "get_session_id",
    "set_user_id",
    "get_user_id",
    "set_request_start_time",
    "get_request_start_time",
    "get_request_duration_ms",
    "set_request_attribute",
    "get_request_attribute",
    "get_all_request_attributes",
    "set_request_context",
    "clear_request_context",
    "get_context_dict",
    "request_context",
    "inject_context_to_trace",
    "inject_context_to_log",
]

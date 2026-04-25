"""
Structured Logging Configuration using structlog

Provides JSON-formatted logs with:
- Timestamps in ISO format
- Log levels
- Correlation IDs for request tracking
- Module/function context
- Automatic exception formatting

Usage:
    from observability.logging_config import setup_logging, get_logger

    setup_logging()
    logger = get_logger(__name__)

    logger.info("processing_started", image_path="/path/to/image.dcm", use_rag=True)
    logger.error("processing_failed", error="timeout", duration_ms=5000)
"""

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

import structlog
from structlog.types import Processor

# Context variable for correlation ID (tracks requests across async boundaries)
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

# Context variable for user session tracking
session_id_var: ContextVar[str] = ContextVar("session_id", default="")


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracking."""
    return str(uuid.uuid4())[:8]  # Short ID for readability


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set the correlation ID for the current context.

    Args:
        correlation_id: Optional ID to use. If None, generates a new one.

    Returns:
        The correlation ID that was set.
    """
    cid = correlation_id or generate_correlation_id()
    correlation_id_var.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return correlation_id_var.get()


def set_session_id(session_id: str) -> None:
    """Set the session ID for the current context."""
    session_id_var.set(session_id)


def get_session_id() -> str:
    """Get the current session ID."""
    return session_id_var.get()


def add_correlation_id(logger: logging.Logger, method_name: str, event_dict: dict) -> dict:
    """Processor to add correlation ID to log events."""
    correlation_id = correlation_id_var.get()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id

    session_id = session_id_var.get()
    if session_id:
        event_dict["session_id"] = session_id

    return event_dict


def add_service_context(logger: logging.Logger, method_name: str, event_dict: dict) -> dict:
    """Processor to add service context to log events."""
    event_dict["service"] = os.getenv("OTEL_SERVICE_NAME", "medical-rag")
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    add_timestamps: bool = True,
) -> None:
    """
    Configure structlog for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format - 'json' for production, 'console' for development
        add_timestamps: Whether to add ISO timestamps to logs
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Build processor chain
    processors: list[Processor] = [
        # Add log level to event dict
        structlog.stdlib.add_log_level,
        # Add our custom context
        add_correlation_id,
        add_service_context,
    ]

    # Add timestamp if requested
    if add_timestamps:
        processors.append(structlog.processors.TimeStamper(fmt="iso"))

    # Add exception formatting
    processors.append(structlog.processors.StackInfoRenderer())
    processors.append(structlog.processors.format_exc_info)

    # Add call site info in debug mode
    if numeric_level <= logging.DEBUG:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            )
        )

    # Choose renderer based on format
    if log_format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Console format for development - more readable
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        A structlog BoundLogger instance

    Example:
        logger = get_logger(__name__)
        logger.info("analysis_complete", confidence=0.95, sources=3)
    """
    return structlog.get_logger(name)


# Convenience function for adding context to all subsequent logs
def bind_context(**kwargs) -> None:
    """
    Bind context values to the current logger.

    These values will be included in all subsequent log events.

    Example:
        bind_context(user_id="123", request_type="analysis")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context values."""
    structlog.contextvars.clear_contextvars()

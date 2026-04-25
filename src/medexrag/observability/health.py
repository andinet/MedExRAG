"""
Health Check Module for Medical X-Ray RAG Application

This module provides health check functionality for monitoring the status
of application components (VLM, vector store, observability backends).

Usage:
    from observability.health import HealthChecker, get_health_checker

    # Get singleton health checker
    checker = get_health_checker()

    # Check all components
    status = checker.full_check()
    print(status.to_dict())

    # Check individual components
    vlm_health = checker.check_vlm()
    vector_health = checker.check_vector_store()
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels for components."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    last_checked: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "last_checked": self.last_checked.isoformat(),
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
        }


class HealthChecker:
    """
    Health checker for Medical RAG application components.

    Checks the health of:
    - VLM (Vision Language Model)
    - Vector Store (ChromaDB)
    - Prometheus (metrics backend)
    - Jaeger/OTEL (tracing backend)
    """

    def __init__(self):
        """Initialize the health checker."""
        self._vlm_instance = None
        self._vector_store_instance = None
        self._cache: Dict[str, ComponentHealth] = {}
        self._cache_ttl_seconds = 30  # Cache health checks for 30 seconds
        self._lock = threading.Lock()

    def set_vlm_instance(self, vlm):
        """Set the VLM instance to check."""
        self._vlm_instance = vlm

    def set_vector_store_instance(self, vector_store):
        """Set the vector store instance to check."""
        self._vector_store_instance = vector_store

    def _is_cache_valid(self, component_name: str) -> bool:
        """Check if cached health check is still valid."""
        if component_name not in self._cache:
            return False
        cached = self._cache[component_name]
        age = (datetime.now() - cached.last_checked).total_seconds()
        return age < self._cache_ttl_seconds

    def check_vlm(self, use_cache: bool = True) -> ComponentHealth:
        """
        Check VLM health status.

        Args:
            use_cache: Whether to use cached result if available

        Returns:
            ComponentHealth with VLM status
        """
        if use_cache and self._is_cache_valid("vlm"):
            return self._cache["vlm"]

        start_time = time.time()

        try:
            if self._vlm_instance is None:
                return ComponentHealth(
                    name="vlm",
                    status=HealthStatus.UNKNOWN,
                    message="VLM instance not configured",
                    details={"configured": False},
                )

            # Check if model is loaded
            model_name = getattr(self._vlm_instance, "model_name", "unknown")
            device = getattr(self._vlm_instance, "device", "unknown")

            # Check if model attribute exists (indicates loaded)
            is_loaded = hasattr(self._vlm_instance, "model") and self._vlm_instance.model is not None

            latency_ms = (time.time() - start_time) * 1000

            if is_loaded:
                health = ComponentHealth(
                    name="vlm",
                    status=HealthStatus.HEALTHY,
                    message="VLM model loaded and ready",
                    latency_ms=latency_ms,
                    details={
                        "model": model_name,
                        "device": str(device),
                        "loaded": True,
                    },
                )
            else:
                health = ComponentHealth(
                    name="vlm",
                    status=HealthStatus.UNHEALTHY,
                    message="VLM model not loaded",
                    latency_ms=latency_ms,
                    details={
                        "model": model_name,
                        "loaded": False,
                    },
                )

        except Exception as e:
            logger.error(f"VLM health check failed: {e}")
            health = ComponentHealth(
                name="vlm",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)},
            )

        with self._lock:
            self._cache["vlm"] = health
        return health

    def check_vector_store(self, use_cache: bool = True) -> ComponentHealth:
        """
        Check vector store health status.

        Args:
            use_cache: Whether to use cached result if available

        Returns:
            ComponentHealth with vector store status
        """
        if use_cache and self._is_cache_valid("vector_store"):
            return self._cache["vector_store"]

        start_time = time.time()

        try:
            if self._vector_store_instance is None:
                return ComponentHealth(
                    name="vector_store",
                    status=HealthStatus.UNKNOWN,
                    message="Vector store instance not configured",
                    details={"configured": False},
                )

            # Get chunk count
            # MedicalVectorStore wraps LangChain Chroma as self.vectorstore
            # The underlying ChromaDB collection is at vectorstore._collection
            chunk_count = 0
            vs = getattr(self._vector_store_instance, "vectorstore", None)
            if vs is not None and hasattr(vs, "_collection"):
                chunk_count = vs._collection.count()

            latency_ms = (time.time() - start_time) * 1000

            if chunk_count > 0:
                health = ComponentHealth(
                    name="vector_store",
                    status=HealthStatus.HEALTHY,
                    message=f"Vector store operational with {chunk_count} chunks",
                    latency_ms=latency_ms,
                    details={
                        "chunks": chunk_count,
                        "collection": getattr(self._vector_store_instance, "collection_name", "unknown"),
                    },
                )
            else:
                health = ComponentHealth(
                    name="vector_store",
                    status=HealthStatus.DEGRADED,
                    message="Vector store is empty - RAG will have no literature context",
                    latency_ms=latency_ms,
                    details={
                        "chunks": 0,
                        "warning": "Run literature ingestion to populate knowledge base",
                    },
                )

        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            health = ComponentHealth(
                name="vector_store",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)},
            )

        with self._lock:
            self._cache["vector_store"] = health
        return health

    def check_prometheus(self, use_cache: bool = True) -> ComponentHealth:
        """
        Check Prometheus connectivity.

        Args:
            use_cache: Whether to use cached result if available

        Returns:
            ComponentHealth with Prometheus status
        """
        if use_cache and self._is_cache_valid("prometheus"):
            return self._cache["prometheus"]

        start_time = time.time()

        try:
            import os
            import urllib.request

            # Try to reach Prometheus health endpoint
            prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
            health_url = f"{prometheus_url}/-/healthy"

            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                is_healthy = response.status == 200

            latency_ms = (time.time() - start_time) * 1000

            if is_healthy:
                health = ComponentHealth(
                    name="prometheus",
                    status=HealthStatus.HEALTHY,
                    message="Prometheus is reachable",
                    latency_ms=latency_ms,
                    details={"url": prometheus_url},
                )
            else:
                health = ComponentHealth(
                    name="prometheus",
                    status=HealthStatus.UNHEALTHY,
                    message="Prometheus returned non-200 status",
                    latency_ms=latency_ms,
                    details={"url": prometheus_url},
                )

        except Exception as e:
            # Prometheus not available - this is OK, app should still work
            health = ComponentHealth(
                name="prometheus",
                status=HealthStatus.DEGRADED,
                message="Prometheus not reachable (metrics collection disabled)",
                latency_ms=(time.time() - start_time) * 1000,
                details={"error": str(e), "note": "Application will continue without metrics"},
            )

        with self._lock:
            self._cache["prometheus"] = health
        return health

    def check_jaeger(self, use_cache: bool = True) -> ComponentHealth:
        """
        Check Jaeger/OTEL collector connectivity.

        Args:
            use_cache: Whether to use cached result if available

        Returns:
            ComponentHealth with Jaeger status
        """
        if use_cache and self._is_cache_valid("jaeger"):
            return self._cache["jaeger"]

        start_time = time.time()

        try:
            import os
            import urllib.request

            # Try to reach Jaeger UI
            jaeger_url = os.getenv("JAEGER_URL", "http://jaeger:16686")

            req = urllib.request.Request(jaeger_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                is_healthy = response.status == 200

            latency_ms = (time.time() - start_time) * 1000

            if is_healthy:
                health = ComponentHealth(
                    name="jaeger",
                    status=HealthStatus.HEALTHY,
                    message="Jaeger is reachable",
                    latency_ms=latency_ms,
                    details={"url": jaeger_url},
                )
            else:
                health = ComponentHealth(
                    name="jaeger",
                    status=HealthStatus.UNHEALTHY,
                    message="Jaeger returned non-200 status",
                    latency_ms=latency_ms,
                    details={"url": jaeger_url},
                )

        except Exception as e:
            # Jaeger not available - this is OK, app should still work
            health = ComponentHealth(
                name="jaeger",
                status=HealthStatus.DEGRADED,
                message="Jaeger not reachable (distributed tracing disabled)",
                latency_ms=(time.time() - start_time) * 1000,
                details={"error": str(e), "note": "Application will continue without tracing"},
            )

        with self._lock:
            self._cache["jaeger"] = health
        return health

    def full_check(self, use_cache: bool = True) -> SystemHealth:
        """
        Perform a full health check of all components.

        Args:
            use_cache: Whether to use cached results if available

        Returns:
            SystemHealth with overall status and component details
        """
        components = {
            "vlm": self.check_vlm(use_cache),
            "vector_store": self.check_vector_store(use_cache),
            "prometheus": self.check_prometheus(use_cache),
            "jaeger": self.check_jaeger(use_cache),
        }

        # Determine overall status
        statuses = [comp.status for comp in components.values()]

        if HealthStatus.UNHEALTHY in statuses:
            # Check if critical components (VLM) are unhealthy
            if components["vlm"].status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            else:
                overall_status = HealthStatus.DEGRADED
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        return SystemHealth(
            status=overall_status,
            components=components,
        )

    def get_health_summary(self) -> Dict[str, str]:
        """
        Get a simple health summary suitable for display.

        Returns:
            Dict with component names and their status strings
        """
        health = self.full_check()
        return {
            "overall": health.status.value,
            **{name: comp.status.value for name, comp in health.components.items()},
        }


# Singleton instance
_health_checker: Optional[HealthChecker] = None
_health_checker_lock = threading.Lock()


def get_health_checker() -> HealthChecker:
    """Get the singleton health checker instance."""
    global _health_checker
    with _health_checker_lock:
        if _health_checker is None:
            _health_checker = HealthChecker()
        return _health_checker


def configure_health_checker(vlm=None, vector_store=None):
    """
    Configure the health checker with application instances.

    Args:
        vlm: VLM instance to monitor
        vector_store: Vector store instance to monitor
    """
    checker = get_health_checker()
    if vlm is not None:
        checker.set_vlm_instance(vlm)
    if vector_store is not None:
        checker.set_vector_store_instance(vector_store)

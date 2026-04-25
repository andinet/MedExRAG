"""
LangSmith Configuration for LLM/Agent Tracing

Provides integration with LangSmith for:
- VLM inference tracing (custom runs for non-LangChain models)
- LangChain/LangGraph automatic tracing
- Run metadata and feedback collection

Setup:
    1. Get API key from https://smith.langchain.com/
    2. Set environment variables:
       LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxx
       LANGCHAIN_TRACING_V2=true
       LANGCHAIN_PROJECT=medical-xray-rag

Usage:
    from observability.langsmith_config import setup_langsmith, trace_vlm_call

    setup_langsmith()

    # For custom VLM calls (non-LangChain)
    with trace_vlm_call("analyze_xray", inputs={"prompt": "..."}) as run:
        result = vlm.generate(image, prompt)
        run.end(outputs={"response": result})
"""

import functools
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

# LangSmith imports
try:
    from langchain_core.tracers import LangChainTracer
    from langsmith import Client
    from langsmith.run_trees import RunTree

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None
    RunTree = None
    LangChainTracer = None


# Module-level client instance
_client: Optional["Client"] = None
_tracing_enabled: bool = False


@dataclass
class RunContext:
    """Context for a traced run."""

    run_id: str
    name: str
    start_time: float
    inputs: Dict[str, Any]
    run_tree: Optional["RunTree"] = None

    def end(
        self,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End the run with outputs or error."""
        if self.run_tree:
            end_time = datetime.now(timezone.utc)
            if error:
                self.run_tree.end(error=error, end_time=end_time)
            else:
                self.run_tree.end(outputs=outputs or {}, end_time=end_time)
            self.run_tree.post()


def setup_langsmith(
    api_key: Optional[str] = None, project_name: Optional[str] = None, endpoint: Optional[str] = None
) -> bool:
    """
    Configure LangSmith tracing.

    Args:
        api_key: LangSmith API key (defaults to LANGCHAIN_API_KEY env var)
        project_name: Project name (defaults to LANGCHAIN_PROJECT env var)
        endpoint: API endpoint (defaults to LANGCHAIN_ENDPOINT env var)

    Returns:
        True if LangSmith was configured successfully
    """
    global _client, _tracing_enabled

    if not LANGSMITH_AVAILABLE:
        print("Warning: langsmith package not installed. Tracing disabled.")
        return False

    # Get configuration from args or environment
    api_key = api_key or os.getenv("LANGCHAIN_API_KEY")
    project_name = project_name or os.getenv("LANGCHAIN_PROJECT", "medical-xray-rag")
    endpoint = endpoint or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    if not api_key:
        print("Warning: LANGCHAIN_API_KEY not set. LangSmith tracing disabled.")
        return False

    # Set environment variables for LangChain automatic tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    try:
        _client = Client(api_key=api_key, api_url=endpoint)
        _tracing_enabled = True
        print(f"LangSmith tracing enabled for project: {project_name}")
        return True
    except Exception as e:
        print(f"Warning: Failed to initialize LangSmith client: {e}")
        return False


def get_langsmith_client() -> Optional["Client"]:
    """Get the LangSmith client instance."""
    return _client


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    return _tracing_enabled


def create_langsmith_callback(
    project_name: Optional[str] = None, tags: Optional[list] = None
) -> Optional["LangChainTracer"]:
    """
    Create a LangSmith callback handler for LangChain operations.

    Args:
        project_name: Override project name
        tags: Optional tags for the trace

    Returns:
        LangChainTracer instance or None if tracing disabled
    """
    if not _tracing_enabled or not LANGSMITH_AVAILABLE:
        return None

    project = project_name or os.getenv("LANGCHAIN_PROJECT", "medical-xray-rag")
    return LangChainTracer(project_name=project, tags=tags or [])


@contextmanager
def trace_vlm_call(
    name: str,
    inputs: Dict[str, Any],
    run_type: str = "llm",
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    parent_run: Optional[RunContext] = None,
):
    """
    Context manager for tracing VLM inference calls.

    This is used for custom VLM calls that aren't using LangChain,
    such as direct Qwen2-VL inference.

    Args:
        name: Name of the operation (e.g., "vlm_generate", "initial_analysis")
        inputs: Input parameters to log
        run_type: Type of run ("llm", "chain", "tool", "retriever")
        metadata: Additional metadata
        tags: Tags for filtering in LangSmith UI
        parent_run: Optional parent run for nested tracing

    Yields:
        RunContext with run_id and end() method

    Example:
        with trace_vlm_call("analyze_xray", {"prompt": prompt}) as run:
            result = vlm.generate(image, prompt)
            run.end(outputs={"response": result, "tokens": 512})
    """
    if not _tracing_enabled or not LANGSMITH_AVAILABLE:
        # Yield a dummy context that does nothing
        yield RunContext(run_id="disabled", name=name, start_time=time.time(), inputs=inputs)
        return

    project_name = os.getenv("LANGCHAIN_PROJECT", "medical-xray-rag")
    start_time = datetime.now(timezone.utc)

    try:
        # Create run tree for hierarchical tracing
        run_tree = RunTree(
            name=name,
            run_type=run_type,
            inputs=inputs,
            project_name=project_name,
            start_time=start_time,
            extra=metadata or {},
            tags=tags or [],
        )

        context = RunContext(
            run_id=str(run_tree.id), name=name, start_time=time.time(), inputs=inputs, run_tree=run_tree
        )

        yield context

    except Exception as e:
        # If tracing fails, still yield a context but log the error
        print(f"Warning: LangSmith tracing error: {e}")
        yield RunContext(run_id="error", name=name, start_time=time.time(), inputs=inputs)


@contextmanager
def trace_pipeline(name: str, inputs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager for tracing full pipeline operations.

    Creates a parent run that can contain nested child runs.

    Args:
        name: Pipeline name (e.g., "analyze_xray_pipeline")
        inputs: Pipeline inputs
        metadata: Additional metadata

    Example:
        with trace_pipeline("analyze_xray", {"image": path, "use_rag": True}) as pipeline:
            # Child operations will be nested under this
            with trace_vlm_call("initial_analysis", {...}) as child:
                ...
    """
    with trace_vlm_call(name=name, inputs=inputs, run_type="chain", metadata=metadata, tags=["pipeline"]) as context:
        yield context


def trace_function(
    name: Optional[str] = None,
    run_type: str = "chain",
    capture_inputs: bool = True,
    capture_outputs: bool = True,
    tags: Optional[list] = None,
) -> Callable:
    """
    Decorator for tracing function calls.

    Args:
        name: Override function name in traces
        run_type: Type of run
        capture_inputs: Whether to capture function arguments
        capture_outputs: Whether to capture return value
        tags: Tags for the trace

    Example:
        @trace_function(name="search_literature", tags=["rag"])
        def search(query: str, k: int = 5):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trace_name = name or func.__name__

            # Capture inputs if requested
            inputs = {}
            if capture_inputs:
                # Get argument names
                import inspect

                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())

                # Map positional args to names
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        # Skip 'self' for methods
                        if param_names[i] != "self":
                            inputs[param_names[i]] = _safe_serialize(arg)

                # Add keyword args
                for k, v in kwargs.items():
                    inputs[k] = _safe_serialize(v)

            with trace_vlm_call(trace_name, inputs=inputs, run_type=run_type, tags=tags) as run:
                try:
                    result = func(*args, **kwargs)

                    if capture_outputs:
                        run.end(outputs={"result": _safe_serialize(result)})
                    else:
                        run.end(outputs={})

                    return result
                except Exception as e:
                    run.end(error=str(e))
                    raise

        return wrapper

    return decorator


def _safe_serialize(value: Any) -> Any:
    """Safely serialize a value for logging (handle non-JSON-serializable types)."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value[:10]]  # Limit list size
    if isinstance(value, dict):
        return {k: _safe_serialize(v) for k, v in list(value.items())[:20]}
    # For images, paths, etc. - just return type info
    return f"<{type(value).__name__}>"


def add_feedback(run_id: str, key: str, score: float, comment: Optional[str] = None) -> bool:
    """
    Add feedback to a LangSmith run.

    Args:
        run_id: The run ID to add feedback to
        key: Feedback key (e.g., "accuracy", "helpfulness")
        score: Score value (0.0 to 1.0)
        comment: Optional comment

    Returns:
        True if feedback was added successfully
    """
    if not _client:
        return False

    try:
        _client.create_feedback(run_id=run_id, key=key, score=score, comment=comment)
        return True
    except Exception as e:
        print(f"Warning: Failed to add feedback: {e}")
        return False

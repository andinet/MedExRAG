"""
MLflow Integration for Experiment Tracking

This module integrates MLflow for tracking experiments, metrics, and artifacts
in the multi-agent RAG system.

MLflow provides:
    - Experiment tracking (parameters, metrics, artifacts)
    - Model registry (version and stage models)
    - Run comparison (compare different configurations)
    - Artifact storage (save evaluation results, models)

What We Track:
    1. RAG Configuration (k, chunk_size, model_name)
    2. Evaluation Metrics (precision, recall, scores)
    3. Agent Performance (latency, quality scores)
    4. Artifacts (reports, benchmark results)

Usage:
    from evaluation.mlflow_tracking import MLflowTracker

    # Initialize tracker
    tracker = MLflowTracker(experiment_name="medical-rag-experiments")

    # Track a RAG experiment
    with tracker.start_run(run_name="rag_k5_chunk800"):
        tracker.log_rag_config(k=5, chunk_size=800)
        tracker.log_retrieval_metrics(precision=0.8, recall=0.6)
        tracker.log_agent_metrics("analyst", score=0.85, latency_ms=1200)
        tracker.log_artifact("results.json")

    # Compare runs
    runs = tracker.get_experiment_runs()

Setup:
    pip install mlflow

    # Start MLflow UI (optional)
    mlflow ui --port 5000

References:
    - MLflow Docs: https://mlflow.org/docs/latest/index.html
    - MLflow Tracking: https://mlflow.org/docs/latest/tracking.html
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# MLflow import with fallback
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None


@dataclass
class RAGConfig:
    """Configuration for a RAG experiment."""

    k_sources: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 150
    embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    score_threshold: float = 0.6
    use_rag: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentMetrics:
    """Metrics from an experiment run."""

    # Retrieval metrics
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_k: Optional[float] = None

    # Agent metrics
    analyst_score: Optional[float] = None
    researcher_score: Optional[float] = None
    diagnostician_score: Optional[float] = None
    reporter_score: Optional[float] = None

    # E2E metrics
    overall_score: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    benchmark_pass_rate: Optional[float] = None

    # LLM Judge metrics
    faithfulness: Optional[float] = None
    relevance: Optional[float] = None
    coherence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class MLflowTracker:
    """
    MLflow experiment tracker for the multi-agent RAG system.

    This class wraps MLflow functionality to provide easy tracking of
    RAG experiments, configurations, and evaluation metrics.

    Example:
        tracker = MLflowTracker(experiment_name="rag-experiments")

        # Start a run
        with tracker.start_run(run_name="experiment_001"):
            # Log configuration
            tracker.log_rag_config(k=5, chunk_size=800)

            # Run evaluation
            metrics = run_evaluation()

            # Log metrics
            tracker.log_metrics(metrics)

            # Log artifacts
            tracker.log_artifact("evaluation_report.json")

        # Get best run
        best = tracker.get_best_run(metric="overall_score")
    """

    def __init__(
        self,
        experiment_name: str = "medical-rag-experiments",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            artifact_location: Where to store artifacts
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
        self.artifact_location = artifact_location
        self._active_run = None

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.client = MlflowClient(self.tracking_uri)
        else:
            print("Warning: MLflow not installed. Tracking disabled.")
            print("Install with: pip install mlflow")
            self.client = None

    @property
    def is_available(self) -> bool:
        """Check if MLflow is available."""
        return MLFLOW_AVAILABLE

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None, description: Optional[str] = None
    ):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Optional tags for the run
            description: Optional description

        Returns:
            Context manager for the run
        """
        if not MLFLOW_AVAILABLE:
            return _MockRunContext()

        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return mlflow.start_run(run_name=run_name, tags=tags, description=description)

    def log_rag_config(self, config: Optional[RAGConfig] = None, **kwargs) -> None:
        """
        Log RAG configuration parameters.

        Args:
            config: RAGConfig object
            **kwargs: Individual config parameters
        """
        if not MLFLOW_AVAILABLE:
            return

        if config:
            params = config.to_dict()
        else:
            params = kwargs

        # Log each parameter
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_retrieval_metrics(
        self,
        precision_at_k: Optional[float] = None,
        recall_at_k: Optional[float] = None,
        mrr: Optional[float] = None,
        ndcg_at_k: Optional[float] = None,
        hit_rate: Optional[float] = None,
        k: int = 5,
    ) -> None:
        """
        Log retrieval evaluation metrics.

        Args:
            precision_at_k: Precision@k
            recall_at_k: Recall@k
            mrr: Mean Reciprocal Rank
            ndcg_at_k: NDCG@k
            hit_rate: Hit rate@k
            k: The k value used
        """
        if not MLFLOW_AVAILABLE:
            return

        metrics = {
            f"precision_at_{k}": precision_at_k,
            f"recall_at_{k}": recall_at_k,
            "mrr": mrr,
            f"ndcg_at_{k}": ndcg_at_k,
            f"hit_rate_at_{k}": hit_rate,
        }

        for name, value in metrics.items():
            if value is not None:
                mlflow.log_metric(name, value)

    def log_agent_metrics(
        self, agent_name: str, score: float, latency_ms: Optional[float] = None, **additional_metrics
    ) -> None:
        """
        Log metrics for a specific agent.

        Args:
            agent_name: Name of the agent (analyst, researcher, etc.)
            score: Overall quality score (0-1)
            latency_ms: Execution time in milliseconds
            **additional_metrics: Additional metrics to log
        """
        if not MLFLOW_AVAILABLE:
            return

        mlflow.log_metric(f"{agent_name}_score", score)

        if latency_ms is not None:
            mlflow.log_metric(f"{agent_name}_latency_ms", latency_ms)

        for metric_name, value in additional_metrics.items():
            mlflow.log_metric(f"{agent_name}_{metric_name}", value)

    def log_e2e_metrics(
        self,
        overall_score: float,
        latency_ms: float,
        pass_rate: Optional[float] = None,
        num_tests: Optional[int] = None,
    ) -> None:
        """
        Log end-to-end benchmark metrics.

        Args:
            overall_score: Overall quality score
            latency_ms: Total pipeline latency
            pass_rate: Benchmark pass rate
            num_tests: Number of tests run
        """
        if not MLFLOW_AVAILABLE:
            return

        mlflow.log_metric("e2e_overall_score", overall_score)
        mlflow.log_metric("e2e_latency_ms", latency_ms)

        if pass_rate is not None:
            mlflow.log_metric("e2e_pass_rate", pass_rate)
        if num_tests is not None:
            mlflow.log_metric("e2e_num_tests", num_tests)

    def log_llm_judge_metrics(
        self,
        faithfulness: Optional[float] = None,
        relevance: Optional[float] = None,
        coherence: Optional[float] = None,
        clinical_appropriateness: Optional[float] = None,
        overall: Optional[float] = None,
    ) -> None:
        """
        Log LLM-as-judge evaluation metrics.

        Args:
            faithfulness: Faithfulness score (0-1)
            relevance: Relevance score (0-1)
            coherence: Coherence score (0-1)
            clinical_appropriateness: Clinical appropriateness score
            overall: Overall LLM judge score
        """
        if not MLFLOW_AVAILABLE:
            return

        metrics = {
            "judge_faithfulness": faithfulness,
            "judge_relevance": relevance,
            "judge_coherence": coherence,
            "judge_clinical": clinical_appropriateness,
            "judge_overall": overall,
        }

        for name, value in metrics.items():
            if value is not None:
                mlflow.log_metric(name, value)

    def log_metrics(self, metrics: ExperimentMetrics) -> None:
        """
        Log all metrics from an ExperimentMetrics object.

        Args:
            metrics: ExperimentMetrics object
        """
        if not MLFLOW_AVAILABLE:
            return

        for name, value in metrics.to_dict().items():
            mlflow.log_metric(name, value)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact (file or directory).

        Args:
            path: Local path to artifact
            artifact_path: Destination path in artifact store
        """
        if not MLFLOW_AVAILABLE:
            return

        if Path(path).exists():
            mlflow.log_artifact(path, artifact_path)
        else:
            print(f"Warning: Artifact not found: {path}")

    def log_dict_as_artifact(self, data: Dict[str, Any], filename: str, artifact_path: Optional[str] = None) -> None:
        """
        Log a dictionary as a JSON artifact.

        Args:
            data: Dictionary to log
            filename: Name for the JSON file
            artifact_path: Destination path
        """
        if not MLFLOW_AVAILABLE:
            return

        # Write to temp file then log
        temp_path = Path(tempfile.gettempdir()) / filename
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        mlflow.log_artifact(str(temp_path), artifact_path)
        temp_path.unlink()  # Clean up

    def get_experiment_runs(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Get all runs from the current experiment.

        Args:
            max_results: Maximum number of runs to return

        Returns:
            List of run dictionaries
        """
        if not MLFLOW_AVAILABLE or not self.client:
            return []

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return []

        runs = self.client.search_runs(experiment_ids=[experiment.experiment_id], max_results=max_results)

        return [
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
            }
            for run in runs
        ]

    def get_best_run(self, metric: str = "e2e_overall_score", maximize: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric name to optimize
            maximize: Whether to maximize (True) or minimize (False)

        Returns:
            Best run dictionary or None
        """
        runs = self.get_experiment_runs()
        if not runs:
            return None

        # Filter runs that have the metric
        valid_runs = [r for r in runs if metric in r["metrics"]]
        if not valid_runs:
            return None

        # Sort by metric
        sorted_runs = sorted(valid_runs, key=lambda r: r["metrics"][metric], reverse=maximize)

        return sorted_runs[0]

    def compare_runs(self, run_ids: List[str], metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: Metrics to include (default: all)

        Returns:
            Dictionary mapping run_id to metrics
        """
        if not MLFLOW_AVAILABLE or not self.client:
            return {}

        comparison = {}
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                run_metrics = run.data.metrics

                if metrics:
                    run_metrics = {k: v for k, v in run_metrics.items() if k in metrics}

                comparison[run_id] = {"run_name": run.info.run_name, "metrics": run_metrics, "params": run.data.params}
            except Exception as e:
                comparison[run_id] = {"error": str(e)}

        return comparison


class _MockRunContext:
    """Mock context manager when MLflow is not available."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# =============================================================================
# Convenience Functions
# =============================================================================


def track_rag_experiment(
    config: RAGConfig,
    retrieval_metrics: Dict[str, float],
    agent_metrics: Dict[str, Dict[str, float]],
    e2e_metrics: Dict[str, float],
    run_name: Optional[str] = None,
    experiment_name: str = "medical-rag-experiments",
) -> Optional[str]:
    """
    Convenience function to track a complete RAG experiment.

    Args:
        config: RAG configuration
        retrieval_metrics: Retrieval metrics dict
        agent_metrics: Per-agent metrics dict
        e2e_metrics: End-to-end metrics dict
        run_name: Optional run name
        experiment_name: MLflow experiment name

    Returns:
        Run ID if successful, None otherwise
    """
    tracker = MLflowTracker(experiment_name=experiment_name)

    if not tracker.is_available:
        return None

    with tracker.start_run(run_name=run_name) as run:
        # Log config
        tracker.log_rag_config(config)

        # Log retrieval metrics
        tracker.log_retrieval_metrics(**retrieval_metrics)

        # Log agent metrics
        for agent_name, metrics in agent_metrics.items():
            tracker.log_agent_metrics(agent_name, **metrics)

        # Log E2E metrics
        tracker.log_e2e_metrics(**e2e_metrics)

        return run.info.run_id


# =============================================================================
# CLI for demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MLflow Tracking Demo")
    print("=" * 60)

    tracker = MLflowTracker(experiment_name="demo-experiment")

    if not tracker.is_available:
        print("\nMLflow not installed. Install with: pip install mlflow")
        print("This demo shows what WOULD be tracked:\n")

    # Demo: Track an experiment
    print("\n1. Starting experiment run...")

    config = RAGConfig(k_sources=5, chunk_size=800, vlm_model="Qwen/Qwen2-VL-2B-Instruct")

    print(f"   Config: k={config.k_sources}, chunk_size={config.chunk_size}")

    # Simulated metrics
    retrieval_metrics = {"precision_at_k": 0.75, "recall_at_k": 0.60, "mrr": 0.82, "ndcg_at_k": 0.71}

    agent_metrics = {
        "analyst": {"score": 0.85, "latency_ms": 1200},
        "researcher": {"score": 0.78, "latency_ms": 300},
        "diagnostician": {"score": 0.82, "latency_ms": 800},
        "reporter": {"score": 0.88, "latency_ms": 600},
    }

    e2e_metrics = {"overall_score": 0.83, "latency_ms": 2900, "pass_rate": 0.90}

    print("\n2. Metrics to be logged:")
    print(f"   Retrieval: {retrieval_metrics}")
    print(f"   Agent scores: {[(k, v['score']) for k, v in agent_metrics.items()]}")
    print(f"   E2E: {e2e_metrics}")

    if tracker.is_available:
        with tracker.start_run(run_name="demo_run"):
            tracker.log_rag_config(config)
            tracker.log_retrieval_metrics(**retrieval_metrics)
            for name, metrics in agent_metrics.items():
                tracker.log_agent_metrics(name, **metrics)
            tracker.log_e2e_metrics(**e2e_metrics)

        print("\n3. Run logged to MLflow!")
        print(f"   Tracking URI: {tracker.tracking_uri}")
        print("   View with: mlflow ui --port 5000")
    else:
        print("\n3. (Skipped - MLflow not installed)")

    print("\n" + "=" * 60)

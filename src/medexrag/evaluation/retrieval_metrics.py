"""
Retrieval Evaluation Metrics for RAG Systems

This module implements standard Information Retrieval metrics for evaluating
the retrieval component of RAG pipelines:

- Precision@k: Fraction of retrieved documents that are relevant
- Recall@k: Fraction of relevant documents that were retrieved
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant document
- NDCG@k (Normalized Discounted Cumulative Gain): Ranking quality metric
- Hit Rate@k: Whether any relevant document appears in top-k

These metrics are essential for evaluating the "Researcher Agent" in our
multi-agent workflow, which performs literature retrieval.

Usage:
    from evaluation.retrieval_metrics import RetrievalEvaluator

    evaluator = RetrievalEvaluator()

    # Single query evaluation
    metrics = evaluator.evaluate_single(
        retrieved_ids=["doc1", "doc2", "doc3"],
        relevant_ids=["doc1", "doc3", "doc5"],
        k=5
    )

    # Batch evaluation
    results = evaluator.evaluate_batch(queries, retrieved, relevant)

References:
    - https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)
    - Ragas: https://docs.ragas.io/en/latest/concepts/metrics/
"""

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_k: float = 0.0  # Normalized Discounted Cumulative Gain
    hit_rate_at_k: float = 0.0  # Whether any relevant doc in top-k
    k: int = 5
    num_retrieved: int = 0
    num_relevant: int = 0
    num_relevant_retrieved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"RetrievalMetrics(P@{self.k}={self.precision_at_k:.3f}, "
            f"R@{self.k}={self.recall_at_k:.3f}, "
            f"MRR={self.mrr:.3f}, "
            f"NDCG@{self.k}={self.ndcg_at_k:.3f}, "
            f"Hit@{self.k}={self.hit_rate_at_k:.3f})"
        )


class RetrievalEvaluator:
    """
    Evaluator for RAG retrieval quality.

    This class computes standard IR metrics to assess how well the
    retrieval component finds relevant documents.

    In the context of our medical RAG system:
    - Query: Initial X-ray findings from Analyst Agent
    - Retrieved: Documents returned by Researcher Agent
    - Relevant: Ground truth relevant documents (from test set)

    Example:
        evaluator = RetrievalEvaluator()

        # Evaluate single query
        metrics = evaluator.evaluate_single(
            retrieved_ids=["pubmed_123", "textbook_456"],
            relevant_ids=["pubmed_123", "guideline_789"],
            k=5
        )
        print(f"Precision@5: {metrics.precision_at_k}")
    """

    def __init__(self, default_k: int = 5):
        """
        Initialize evaluator.

        Args:
            default_k: Default cutoff for @k metrics
        """
        self.default_k = default_k

    def precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Compute Precision@k.

        Precision@k = |relevant ∩ retrieved@k| / k

        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs
            k: Cutoff

        Returns:
            Precision@k score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = retrieved_at_k & relevant

        return len(relevant_retrieved) / k

    def recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Compute Recall@k.

        Recall@k = |relevant ∩ retrieved@k| / |relevant|

        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs
            k: Cutoff

        Returns:
            Recall@k score (0.0 to 1.0)
        """
        if len(relevant) == 0:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = retrieved_at_k & relevant

        return len(relevant_retrieved) / len(relevant)

    def reciprocal_rank(self, retrieved: List[str], relevant: Set[str]) -> float:
        """
        Compute Reciprocal Rank.

        RR = 1 / rank of first relevant document

        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: Set of relevant document IDs

        Returns:
            Reciprocal rank (0.0 to 1.0)
        """
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank

        return 0.0  # No relevant document found

    def dcg_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Compute Discounted Cumulative Gain at k.

        DCG@k = Σ (rel_i / log2(i + 1)) for i in 1..k

        Using binary relevance (1 if relevant, 0 otherwise).

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff

        Returns:
            DCG@k score
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            rel = 1.0 if doc_id in relevant else 0.0
            dcg += rel / math.log2(i + 1)

        return dcg

    def ndcg_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k.

        NDCG@k = DCG@k / IDCG@k

        Where IDCG is the DCG of the ideal ranking (all relevant docs first).

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff

        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        dcg = self.dcg_at_k(retrieved, relevant, k)

        # Ideal DCG: all relevant documents ranked first
        num_relevant = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def hit_rate_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Compute Hit Rate at k (binary: 1 if any relevant in top-k, else 0).

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        retrieved_at_k = set(retrieved[:k])
        return 1.0 if (retrieved_at_k & relevant) else 0.0

    def evaluate_single(
        self, retrieved_ids: List[str], relevant_ids: List[str], k: Optional[int] = None
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval for a single query.

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: List of ground truth relevant document IDs
            k: Cutoff for @k metrics (default: self.default_k)

        Returns:
            RetrievalMetrics with all computed metrics
        """
        k = k or self.default_k
        relevant_set = set(relevant_ids)
        retrieved_at_k = set(retrieved_ids[:k])

        return RetrievalMetrics(
            precision_at_k=self.precision_at_k(retrieved_ids, relevant_set, k),
            recall_at_k=self.recall_at_k(retrieved_ids, relevant_set, k),
            mrr=self.reciprocal_rank(retrieved_ids, relevant_set),
            ndcg_at_k=self.ndcg_at_k(retrieved_ids, relevant_set, k),
            hit_rate_at_k=self.hit_rate_at_k(retrieved_ids, relevant_set, k),
            k=k,
            num_retrieved=len(retrieved_ids),
            num_relevant=len(relevant_set),
            num_relevant_retrieved=len(retrieved_at_k & relevant_set),
        )

    def evaluate_batch(self, queries: List[Dict[str, Any]], k: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate retrieval over multiple queries.

        Args:
            queries: List of query dicts with 'retrieved_ids' and 'relevant_ids'
            k: Cutoff for @k metrics

        Returns:
            Dictionary with aggregate metrics (mean, std, per-query)

        Example:
            queries = [
                {"query": "pneumonia", "retrieved_ids": [...], "relevant_ids": [...]},
                {"query": "cardiomegaly", "retrieved_ids": [...], "relevant_ids": [...]}
            ]
            results = evaluator.evaluate_batch(queries)
        """
        k = k or self.default_k
        all_metrics = []

        for query_data in queries:
            metrics = self.evaluate_single(
                retrieved_ids=query_data["retrieved_ids"], relevant_ids=query_data["relevant_ids"], k=k
            )
            all_metrics.append(metrics)

        # Compute aggregates
        if not all_metrics:
            return {"error": "No queries to evaluate"}

        n = len(all_metrics)

        def mean(values):
            return sum(values) / len(values) if values else 0.0

        def std(values, avg):
            if len(values) < 2:
                return 0.0
            variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
            return math.sqrt(variance)

        p_values = [m.precision_at_k for m in all_metrics]
        r_values = [m.recall_at_k for m in all_metrics]
        mrr_values = [m.mrr for m in all_metrics]
        ndcg_values = [m.ndcg_at_k for m in all_metrics]
        hit_values = [m.hit_rate_at_k for m in all_metrics]

        mean_p = mean(p_values)
        mean_r = mean(r_values)
        mean_mrr = mean(mrr_values)
        mean_ndcg = mean(ndcg_values)
        mean_hit = mean(hit_values)

        return {
            "num_queries": n,
            "k": k,
            "aggregate": {
                "precision_at_k": {"mean": mean_p, "std": std(p_values, mean_p)},
                "recall_at_k": {"mean": mean_r, "std": std(r_values, mean_r)},
                "mrr": {"mean": mean_mrr, "std": std(mrr_values, mean_mrr)},
                "ndcg_at_k": {"mean": mean_ndcg, "std": std(ndcg_values, mean_ndcg)},
                "hit_rate_at_k": {"mean": mean_hit, "std": std(hit_values, mean_hit)},
            },
            "per_query": [m.to_dict() for m in all_metrics],
        }

    def load_golden_dataset(self, path: str) -> List[Dict[str, Any]]:
        """
        Load golden evaluation dataset from JSON file.

        Expected format:
        [
            {
                "query_id": "q1",
                "query": "pneumonia consolidation findings",
                "relevant_ids": ["doc1", "doc2", "doc3"]
            },
            ...
        ]

        Args:
            path: Path to JSON file

        Returns:
            List of query dictionaries
        """
        with open(path, "r") as f:
            return json.load(f)

    def save_results(self, results: Dict[str, Any], path: str) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            results: Results dictionary from evaluate_batch
            path: Output path
        """
        with open(path, "w") as f:
            json.dump(results, f, indent=2)


# =============================================================================
# Convenience Functions
# =============================================================================


def evaluate_retrieval(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> RetrievalMetrics:
    """
    Convenience function for single query evaluation.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Cutoff

    Returns:
        RetrievalMetrics
    """
    evaluator = RetrievalEvaluator(default_k=k)
    return evaluator.evaluate_single(retrieved_ids, relevant_ids, k)


# =============================================================================
# CLI for standalone evaluation
# =============================================================================

if __name__ == "__main__":
    # Demo evaluation
    print("=" * 60)
    print("Retrieval Metrics Evaluation Demo")
    print("=" * 60)

    evaluator = RetrievalEvaluator(default_k=5)

    # Example: Evaluate single query
    retrieved = ["doc1", "doc3", "doc5", "doc7", "doc9"]
    relevant = ["doc1", "doc2", "doc3", "doc4"]

    metrics = evaluator.evaluate_single(retrieved, relevant, k=5)

    print("\nSingle Query Evaluation:")
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant:  {relevant}")
    print(f"\n  {metrics}")

    # Example: Batch evaluation
    queries = [
        {
            "query": "pneumonia findings",
            "retrieved_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "relevant_ids": ["doc1", "doc3"],
        },
        {
            "query": "cardiomegaly",
            "retrieved_ids": ["doc6", "doc7", "doc8", "doc9", "doc10"],
            "relevant_ids": ["doc6", "doc8", "doc11"],
        },
        {
            "query": "pleural effusion",
            "retrieved_ids": ["doc11", "doc12", "doc13", "doc14", "doc15"],
            "relevant_ids": ["doc11", "doc12", "doc13"],
        },
    ]

    results = evaluator.evaluate_batch(queries, k=5)

    print("\n" + "=" * 60)
    print("Batch Evaluation Results:")
    print("=" * 60)
    print(f"\nNumber of queries: {results['num_queries']}")
    print(f"k: {results['k']}")
    print("\nAggregate Metrics:")
    for metric, values in results["aggregate"].items():
        print(f"  {metric}: {values['mean']:.3f} (±{values['std']:.3f})")

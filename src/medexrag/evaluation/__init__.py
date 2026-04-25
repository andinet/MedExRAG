"""
Evaluation Framework for Medical X-Ray RAG System

This package provides evaluation and benchmarking tools for:
- RAG retrieval quality (Precision@k, Recall@k, MRR, NDCG)
- Individual agent performance
- End-to-end pipeline benchmarks
- LLM-as-judge evaluation

Usage:
    from evaluation import RetrievalEvaluator, AgentEvaluator, E2EBenchmark

    # Evaluate retrieval
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate(queries, retrieved_docs, relevant_docs)

    # Evaluate agents
    agent_eval = AgentEvaluator()
    scores = agent_eval.evaluate_analyst(findings, ground_truth)
"""

from .agent_evaluators import AgentEvaluator
from .e2e_benchmarks import E2EBenchmark
from .retrieval_metrics import RetrievalEvaluator

__all__ = [
    "RetrievalEvaluator",
    "AgentEvaluator",
    "E2EBenchmark",
]

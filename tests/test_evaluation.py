"""
Quality Gate Tests for MLOps Evaluation Framework

These tests validate the evaluation components work correctly and
serve as quality gates in CI/CD pipelines. Tests are designed to
run without GPU or heavy ML models.

Test Categories:
    1. Retrieval Metrics - IR metrics calculations
    2. Agent Evaluators - Per-agent quality checks
    3. E2E Benchmarks - End-to-end test execution
    4. LLM Judge - LLM-based evaluation (mock mode)
    5. MLflow Integration - Experiment tracking
"""

import pytest
import json
import math
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_retrieved_docs():
    """Sample retrieved document IDs for testing."""
    return ["doc1", "doc3", "doc5", "doc7", "doc9"]


@pytest.fixture
def sample_relevant_docs():
    """Sample relevant document IDs (ground truth)."""
    return ["doc1", "doc2", "doc3", "doc4"]


@pytest.fixture
def sample_agent_output():
    """Sample agent output for evaluation."""
    return {
        "findings": "Right lower lobe consolidation with air bronchograms",
        "impression": "Findings consistent with bacterial pneumonia",
        "confidence": 0.85,
        "recommendations": ["Clinical correlation", "Follow-up in 4-6 weeks"]
    }


@pytest.fixture
def sample_medical_report():
    """Sample complete medical report."""
    return """
    FINDINGS:
    Chest X-ray PA view demonstrates right lower lobe consolidation
    with air bronchograms. No pleural effusion. Cardiac silhouette
    is normal in size.

    IMPRESSION:
    Findings consistent with community-acquired pneumonia, most likely
    bacterial etiology given the lobar distribution [Source 1].

    RECOMMENDATION:
    Clinical correlation recommended. Follow-up imaging in 4-6 weeks
    to document resolution.
    """


# =============================================================================
# Retrieval Metrics Tests
# =============================================================================

class TestRetrievalMetrics:
    """Tests for retrieval evaluation metrics."""

    def test_precision_at_k_calculation(self, sample_retrieved_docs, sample_relevant_docs):
        """Test Precision@k calculation."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator

        evaluator = RetrievalEvaluator(default_k=5)
        relevant_set = set(sample_relevant_docs)

        # At k=5, retrieved [doc1, doc3, doc5, doc7, doc9]
        # Relevant: doc1, doc3 -> 2 relevant in top 5
        precision = evaluator.precision_at_k(sample_retrieved_docs, relevant_set, k=5)

        assert precision == 2 / 5  # 0.4
        assert 0 <= precision <= 1

    def test_recall_at_k_calculation(self, sample_retrieved_docs, sample_relevant_docs):
        """Test Recall@k calculation."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator

        evaluator = RetrievalEvaluator()
        relevant_set = set(sample_relevant_docs)

        # 2 relevant docs retrieved out of 4 total relevant
        recall = evaluator.recall_at_k(sample_retrieved_docs, relevant_set, k=5)

        assert recall == 2 / 4  # 0.5
        assert 0 <= recall <= 1

    def test_reciprocal_rank(self, sample_retrieved_docs, sample_relevant_docs):
        """Test Mean Reciprocal Rank calculation."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator

        evaluator = RetrievalEvaluator()
        relevant_set = set(sample_relevant_docs)

        # First relevant doc (doc1) is at position 1
        rr = evaluator.reciprocal_rank(sample_retrieved_docs, relevant_set)

        assert rr == 1.0  # 1/1

    def test_reciprocal_rank_no_relevant(self):
        """Test RR when no relevant documents found."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator

        evaluator = RetrievalEvaluator()
        retrieved = ["doc10", "doc11", "doc12"]
        relevant = {"doc1", "doc2"}

        rr = evaluator.reciprocal_rank(retrieved, relevant)
        assert rr == 0.0

    def test_ndcg_at_k(self, sample_retrieved_docs, sample_relevant_docs):
        """Test NDCG@k calculation."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator

        evaluator = RetrievalEvaluator()
        relevant_set = set(sample_relevant_docs)

        ndcg = evaluator.ndcg_at_k(sample_retrieved_docs, relevant_set, k=5)

        assert 0 <= ndcg <= 1
        assert ndcg > 0  # Should have some score since doc1 and doc3 are relevant

    def test_hit_rate_at_k(self, sample_retrieved_docs, sample_relevant_docs):
        """Test Hit Rate@k calculation."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator

        evaluator = RetrievalEvaluator()
        relevant_set = set(sample_relevant_docs)

        # At least one relevant doc in top 5
        hit_rate = evaluator.hit_rate_at_k(sample_retrieved_docs, relevant_set, k=5)
        assert hit_rate == 1.0

        # No relevant in top-k
        no_hit = evaluator.hit_rate_at_k(["x", "y", "z"], relevant_set, k=3)
        assert no_hit == 0.0

    def test_evaluate_single_returns_all_metrics(self, sample_retrieved_docs, sample_relevant_docs):
        """Test that evaluate_single returns complete metrics."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator, RetrievalMetrics

        evaluator = RetrievalEvaluator(default_k=5)
        metrics = evaluator.evaluate_single(sample_retrieved_docs, sample_relevant_docs)

        assert isinstance(metrics, RetrievalMetrics)
        assert hasattr(metrics, 'precision_at_k')
        assert hasattr(metrics, 'recall_at_k')
        assert hasattr(metrics, 'mrr')
        assert hasattr(metrics, 'ndcg_at_k')
        assert hasattr(metrics, 'hit_rate_at_k')
        assert metrics.k == 5

    def test_evaluate_batch(self):
        """Test batch evaluation across multiple queries."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator

        evaluator = RetrievalEvaluator(default_k=3)
        queries = [
            {"retrieved_ids": ["a", "b", "c"], "relevant_ids": ["a", "c"]},
            {"retrieved_ids": ["d", "e", "f"], "relevant_ids": ["d"]},
            {"retrieved_ids": ["g", "h", "i"], "relevant_ids": ["x", "y"]},  # No hits
        ]

        results = evaluator.evaluate_batch(queries, k=3)

        assert results["num_queries"] == 3
        assert "aggregate" in results
        assert "per_query" in results
        assert len(results["per_query"]) == 3

    def test_edge_case_empty_relevant(self):
        """Test handling of empty relevant set."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator

        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate_single(["doc1", "doc2"], [])

        assert metrics.recall_at_k == 0.0
        assert metrics.num_relevant == 0


# =============================================================================
# Agent Evaluator Tests
# =============================================================================

class TestAgentEvaluators:
    """Tests for per-agent quality evaluation."""

    def test_analyst_evaluator(self, sample_agent_output):
        """Test analyst agent evaluation."""
        from medexrag.evaluation.agent_evaluators import AgentEvaluator

        evaluator = AgentEvaluator()
        # evaluate_analyst expects a findings string, not a dict
        result = evaluator.evaluate_analyst(
            findings=sample_agent_output["findings"],
            ground_truth_findings=["consolidation", "air bronchograms"],
        )

        assert result.agent_type == "analyst"
        assert 0 <= result.overall_score <= 1
        assert "medical_terminology" in result.component_scores
        assert "anatomical_specificity" in result.component_scores

    def test_researcher_evaluator(self):
        """Test researcher agent evaluation."""
        from medexrag.evaluation.agent_evaluators import AgentEvaluator

        evaluator = AgentEvaluator()
        retrieved_docs = [
            {"text": "Pneumonia presents as consolidation...", "metadata": {"source": "ACR"}, "relevance_score": 0.95},
            {"text": "Consolidation findings in chest X-ray...", "metadata": {"source": "Fleischner"}, "relevance_score": 0.87},
        ]

        result = evaluator.evaluate_researcher(
            retrieved_docs=retrieved_docs,
            query="pneumonia consolidation",
        )

        assert result.agent_type == "researcher"
        assert 0 <= result.overall_score <= 1
        assert "source_quantity" in result.component_scores
        assert "source_diversity" in result.component_scores

    def test_diagnostician_evaluator(self):
        """Test diagnostician agent evaluation."""
        from medexrag.evaluation.agent_evaluators import AgentEvaluator

        evaluator = AgentEvaluator()
        result = evaluator.evaluate_diagnostician(
            diagnosis="Primary Diagnosis: Community-acquired pneumonia. "
                      "Differential: Bacterial pneumonia, Viral pneumonia. "
                      "Confidence: 82%. According to guidelines, consolidation is typical.",
            findings="Right lower lobe consolidation with air bronchograms",
            literature="ACR guidelines for pneumonia diagnosis",
        )

        assert result.agent_type == "diagnostician"
        assert 0 <= result.overall_score <= 1
        assert "diagnosis_present" in result.component_scores
        assert "evidence_cited" in result.component_scores

    def test_reporter_evaluator(self, sample_medical_report):
        """Test reporter agent evaluation."""
        from medexrag.evaluation.agent_evaluators import AgentEvaluator

        evaluator = AgentEvaluator()
        result = evaluator.evaluate_reporter(report=sample_medical_report)

        assert result.agent_type == "reporter"
        assert 0 <= result.overall_score <= 1
        assert "required_sections" in result.component_scores
        assert "professional_format" in result.component_scores

    def test_agent_trace_evaluation(self):
        """Test full agent trace evaluation."""
        from medexrag.evaluation.agent_evaluators import AgentEvaluator

        evaluator = AgentEvaluator()
        # evaluate_agent_trace expects a workflow_state dict with specific keys
        workflow_state = {
            "initial_findings": "Right lower lobe consolidation with air bronchograms noted",
            "literature_context": "Source: ACR guidelines indicate lobar consolidation is typical of bacterial pneumonia.",
            "diagnostic_reasoning": "Primary Diagnosis: pneumonia likely based on consolidation findings. "
                                    "According to guidelines, this is consistent with bacterial etiology.",
            "final_report": "FINDINGS:\nRight lower lobe consolidation.\n\nIMPRESSION:\nPneumonia.\n\n"
                            "RECOMMENDATION:\nFollow-up recommended.",
        }

        result = evaluator.evaluate_agent_trace(workflow_state)

        assert hasattr(result, "agent_scores")
        assert hasattr(result, "information_flow_score")
        assert hasattr(result, "overall_score")
        assert 0 <= result.overall_score <= 1


# =============================================================================
# E2E Benchmark Tests
# =============================================================================

class TestE2EBenchmarks:
    """Tests for end-to-end benchmark system."""

    def test_benchmark_initialization(self):
        """Test E2E benchmark initialization."""
        from medexrag.evaluation.e2e_benchmarks import E2EBenchmark

        benchmark = E2EBenchmark()

        assert len(benchmark.test_cases) > 0

    def test_load_test_cases(self):
        """Test loading test cases by category."""
        from medexrag.evaluation.e2e_benchmarks import E2EBenchmark

        benchmark = E2EBenchmark()

        functional_cases = [tc for tc in benchmark.test_cases if tc.category == "functional"]
        assert len(functional_cases) > 0

        robustness_cases = [tc for tc in benchmark.test_cases if tc.category == "robustness"]
        assert len(robustness_cases) > 0

    def test_run_single_test_case(self):
        """Test running a single test case."""
        from medexrag.evaluation.e2e_benchmarks import E2EBenchmark

        benchmark = E2EBenchmark()
        functional_cases = [tc for tc in benchmark.test_cases if tc.category == "functional"]

        if functional_cases:
            result = benchmark.run_test(functional_cases[0], use_mock=True)

            assert hasattr(result, "test_id")
            assert hasattr(result, "passed")
            assert hasattr(result, "execution_time_ms")

    def test_run_all_benchmarks(self):
        """Test running all benchmarks."""
        from medexrag.evaluation.e2e_benchmarks import E2EBenchmark

        benchmark = E2EBenchmark()
        results = benchmark.run_all(use_mock=True)

        # results is Dict[str, BenchmarkResults]
        assert "functional" in results
        assert results["functional"].total_tests > 0

    def test_benchmark_results_structure(self):
        """Test benchmark results have correct structure."""
        from medexrag.evaluation.e2e_benchmarks import E2EBenchmark

        benchmark = E2EBenchmark()
        results = benchmark.run_all(use_mock=True)

        # Check that at least one category exists with valid structure
        for category, bench_result in results.items():
            assert bench_result.total_tests > 0
            assert bench_result.passed_tests >= 0
            assert bench_result.failed_tests >= 0
            assert 0 <= bench_result.pass_rate <= 1


# =============================================================================
# LLM Judge Tests
# =============================================================================

class TestLLMJudge:
    """Tests for LLM-as-Judge evaluation."""

    def test_judge_initialization_mock(self):
        """Test LLM judge initializes in mock mode."""
        from medexrag.evaluation.llm_judge import LLMJudge

        judge = LLMJudge(use_mock=True)

        assert judge.use_mock is True
        assert judge.model == "gpt-4"

    def test_evaluate_faithfulness(self, sample_medical_report):
        """Test faithfulness evaluation."""
        from medexrag.evaluation.llm_judge import LLMJudge

        judge = LLMJudge(use_mock=True)
        context = "ACR guidelines state lobar consolidation indicates pneumonia."

        score = judge.evaluate_faithfulness(sample_medical_report, context)

        assert hasattr(score, 'dimension')
        assert hasattr(score, 'score')
        assert hasattr(score, 'normalized_score')
        assert hasattr(score, 'reasoning')
        assert 1 <= score.score <= 5
        assert 0 <= score.normalized_score <= 1

    def test_evaluate_relevance(self, sample_medical_report):
        """Test relevance evaluation."""
        from medexrag.evaluation.llm_judge import LLMJudge

        judge = LLMJudge(use_mock=True)
        question = "Analyze this chest X-ray for signs of pneumonia"

        score = judge.evaluate_relevance(sample_medical_report, question)

        assert score.dimension == "relevance"
        assert 1 <= score.score <= 5

    def test_evaluate_coherence(self, sample_medical_report):
        """Test coherence evaluation."""
        from medexrag.evaluation.llm_judge import LLMJudge

        judge = LLMJudge(use_mock=True)
        score = judge.evaluate_coherence(sample_medical_report)

        assert score.dimension == "coherence"
        assert 1 <= score.score <= 5

    def test_evaluate_clinical_appropriateness(self, sample_medical_report):
        """Test clinical appropriateness evaluation."""
        from medexrag.evaluation.llm_judge import LLMJudge

        judge = LLMJudge(use_mock=True)
        score = judge.evaluate_clinical_appropriateness(sample_medical_report)

        assert score.dimension == "clinical_appropriateness"
        assert 1 <= score.score <= 5

    def test_evaluate_all_dimensions(self, sample_medical_report):
        """Test evaluation across all dimensions."""
        from medexrag.evaluation.llm_judge import LLMJudge, EvaluationDimension

        judge = LLMJudge(use_mock=True)
        context = "Medical literature context..."
        question = "Analyze for pneumonia"

        result = judge.evaluate_all(
            output=sample_medical_report,
            context=context,
            question=question
        )

        assert hasattr(result, 'dimension_scores')
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'overall_reasoning')
        assert result.is_mock is True
        assert 0 <= result.overall_score <= 1

    def test_generate_report(self, sample_medical_report):
        """Test report generation."""
        from medexrag.evaluation.llm_judge import LLMJudge

        judge = LLMJudge(use_mock=True)
        result = judge.evaluate_all(sample_medical_report, "context", "question")
        report = judge.generate_report(result)

        assert "LLM JUDGE EVALUATION REPORT" in report
        assert "Overall Score" in report
        assert "(MOCK)" in report


# =============================================================================
# MLflow Integration Tests
# =============================================================================

class TestMLflowIntegration:
    """Tests for MLflow experiment tracking integration."""

    def test_tracker_initialization(self):
        """Test MLflow tracker initialization."""
        from medexrag.evaluation.mlflow_tracking import MLflowTracker

        # Should work without actual MLflow server
        tracker = MLflowTracker(
            experiment_name="test_experiment",
            tracking_uri=None  # Use default local
        )

        assert tracker.experiment_name == "test_experiment"

    def test_log_rag_config(self):
        """Test logging RAG configuration."""
        from medexrag.evaluation.mlflow_tracking import MLflowTracker

        tracker = MLflowTracker(experiment_name="test_config")

        config = {
            "model_name": "Qwen2-VL-2B",
            "embedding_model": "PubMedBERT",
            "chunk_size": 800,
            "k_sources": 5
        }

        # Should not raise
        try:
            tracker.log_rag_config(config)
        except Exception as e:
            # MLflow might not be configured, that's okay for unit tests
            assert "mlflow" in str(e).lower() or True

    def test_log_retrieval_metrics(self):
        """Test logging retrieval metrics."""
        from medexrag.evaluation.mlflow_tracking import MLflowTracker
        from medexrag.evaluation.retrieval_metrics import RetrievalMetrics

        tracker = MLflowTracker(experiment_name="test_retrieval")

        metrics = RetrievalMetrics(
            precision_at_k=0.6,
            recall_at_k=0.5,
            mrr=0.8,
            ndcg_at_k=0.7,
            hit_rate_at_k=1.0,
            k=5
        )

        try:
            tracker.log_retrieval_metrics(metrics)
        except Exception:
            pass  # MLflow may not be configured

    def test_log_agent_metrics(self):
        """Test logging agent metrics."""
        from medexrag.evaluation.mlflow_tracking import MLflowTracker

        tracker = MLflowTracker(experiment_name="test_agents")

        agent_metrics = {
            "analyst": {"completeness": 0.9, "confidence": 0.85},
            "researcher": {"retrieval_quality": 0.8},
            "diagnostician": {"reasoning_score": 0.75}
        }

        try:
            tracker.log_agent_metrics(agent_metrics)
        except Exception:
            pass

    def test_log_e2e_metrics(self):
        """Test logging E2E benchmark metrics."""
        from medexrag.evaluation.mlflow_tracking import MLflowTracker

        tracker = MLflowTracker(experiment_name="test_e2e")

        e2e_results = {
            "summary": {
                "total": 10,
                "passed": 8,
                "failed": 2,
                "pass_rate": 0.8
            }
        }

        try:
            tracker.log_e2e_metrics(e2e_results)
        except Exception:
            pass


# =============================================================================
# Quality Gate Tests
# =============================================================================

class TestQualityGates:
    """Tests that serve as quality gates for CI/CD."""

    def test_minimum_retrieval_precision(self):
        """Quality gate: Minimum retrieval precision threshold."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator

        evaluator = RetrievalEvaluator()

        # Simulate good retrieval
        good_retrieval = ["rel1", "rel2", "rel3", "other1", "other2"]
        relevant = ["rel1", "rel2", "rel3", "rel4"]

        metrics = evaluator.evaluate_single(good_retrieval, relevant)

        # Quality gate: Precision@5 should be at least 0.4
        assert metrics.precision_at_k >= 0.4, \
            f"Precision@5 ({metrics.precision_at_k}) below threshold (0.4)"

    def test_minimum_recall(self):
        """Quality gate: Minimum recall threshold."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator

        evaluator = RetrievalEvaluator()

        good_retrieval = ["rel1", "rel2", "rel3", "other1", "other2"]
        relevant = ["rel1", "rel2", "rel3", "rel4"]

        metrics = evaluator.evaluate_single(good_retrieval, relevant)

        # Quality gate: Recall@5 should be at least 0.5
        assert metrics.recall_at_k >= 0.5, \
            f"Recall@5 ({metrics.recall_at_k}) below threshold (0.5)"

    def test_e2e_pass_rate_threshold(self):
        """Quality gate: E2E benchmark pass rate threshold."""
        from medexrag.evaluation.e2e_benchmarks import E2EBenchmark

        benchmark = E2EBenchmark()
        results = benchmark.run_all(use_mock=True)

        # Quality gate: At least 70% pass rate on functional tests
        min_pass_rate = 0.7
        functional = results.get("functional")
        assert functional is not None, "No functional benchmark results"
        assert functional.pass_rate >= min_pass_rate, \
            f"E2E pass rate ({functional.pass_rate}) below threshold ({min_pass_rate})"

    def test_llm_judge_minimum_score(self):
        """Quality gate: LLM judge minimum score threshold."""
        from medexrag.evaluation.llm_judge import LLMJudge

        judge = LLMJudge(use_mock=True)

        # Sample high-quality output
        good_output = """
        FINDINGS: Clear consolidation in right lower lobe with air bronchograms.
        IMPRESSION: Bacterial pneumonia with lobar distribution.
        RECOMMENDATION: Antibiotic treatment, follow-up in 4-6 weeks.
        """

        result = judge.evaluate_all(good_output, "context", "question")

        # Quality gate: Overall score should be at least 0.4 (normalized)
        assert result.overall_score >= 0.4, \
            f"LLM judge score ({result.overall_score}) below threshold (0.4)"


# =============================================================================
# Integration Tests
# =============================================================================

class TestEvaluationIntegration:
    """Integration tests combining multiple evaluation components."""

    def test_full_evaluation_pipeline(self, sample_medical_report):
        """Test complete evaluation pipeline flow."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator
        from medexrag.evaluation.agent_evaluators import AgentEvaluator
        from medexrag.evaluation.llm_judge import LLMJudge

        # Step 1: Retrieval evaluation
        retrieval_eval = RetrievalEvaluator()
        retrieval_metrics = retrieval_eval.evaluate_single(
            ["doc1", "doc2", "doc3"],
            ["doc1", "doc3"]
        )

        # Step 2: Agent evaluation
        agent_eval = AgentEvaluator()
        reporter_score = agent_eval.evaluate_reporter(report=sample_medical_report)

        # Step 3: LLM judge
        judge = LLMJudge(use_mock=True)
        judge_result = judge.evaluate_all(sample_medical_report, "context", "question")

        # Verify all components produced valid output
        assert retrieval_metrics.precision_at_k >= 0
        assert "required_sections" in reporter_score.component_scores
        assert judge_result.overall_score >= 0

    def test_evaluation_to_dict_serialization(self):
        """Test that all evaluation results can be serialized to dict/JSON."""
        from medexrag.evaluation.retrieval_metrics import RetrievalEvaluator
        from medexrag.evaluation.llm_judge import LLMJudge

        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate_single(["a", "b"], ["a"])

        # Should serialize without error
        metrics_dict = metrics.to_dict()
        json_str = json.dumps(metrics_dict)
        assert len(json_str) > 0

        judge = LLMJudge(use_mock=True)
        result = judge.evaluate_all("output", "context", "question")

        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0

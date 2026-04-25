"""
End-to-End Benchmarks for Multi-Agent RAG System

This module provides end-to-end benchmark tests that evaluate the complete
pipeline from input (X-ray + question) to output (final report).

Unlike component-level evaluation, E2E benchmarks test:
- Real-world scenarios
- System integration
- Performance under various conditions
- Edge cases and failure modes

Benchmark Categories:
    1. Functional Benchmarks - Does the system produce correct outputs?
    2. Performance Benchmarks - How fast and efficient is the system?
    3. Robustness Benchmarks - How does it handle edge cases?
    4. Comparison Benchmarks - RAG vs non-RAG performance

Usage:
    from evaluation.e2e_benchmarks import E2EBenchmark

    benchmark = E2EBenchmark()

    # Run all benchmarks
    results = benchmark.run_all()

    # Run specific benchmark
    results = benchmark.run_functional_benchmarks()
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TestCase:
    """A single test case for E2E benchmarking."""

    id: str
    name: str
    description: str
    category: str  # functional, performance, robustness
    input_data: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    timeout_seconds: float = 60.0
    tags: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of a single test case execution."""

    test_id: str
    test_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    execution_time_ms: float
    actual_outputs: Dict[str, Any]
    checks: Dict[str, bool]
    errors: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResults:
    """Aggregate results from a benchmark suite."""

    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    avg_score: float
    avg_execution_time_ms: float
    test_results: List[TestResult]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "test_results": [r.to_dict() for r in self.test_results]}


class E2EBenchmark:
    """
    End-to-end benchmark suite for the multi-agent RAG system.

    This class manages test cases, executes benchmarks, and reports results.
    It can work with or without actual pipeline execution (mock mode for CI).

    Example:
        benchmark = E2EBenchmark()

        # Run with actual pipeline
        from rag_pipeline import MedicalRAGPipeline
        pipeline = MedicalRAGPipeline(load_vlm=True)
        results = benchmark.run_all(pipeline_func=pipeline.analyze_xray)

        # Run with mock (for CI/testing)
        results = benchmark.run_all(use_mock=True)
    """

    def __init__(self, test_cases_path: Optional[str] = None):
        """
        Initialize benchmark suite.

        Args:
            test_cases_path: Optional path to JSON file with test cases
        """
        self.test_cases: List[TestCase] = []

        if test_cases_path and Path(test_cases_path).exists():
            self.load_test_cases(test_cases_path)
        else:
            self._load_default_test_cases()

    def _load_default_test_cases(self):
        """Load built-in test cases for demonstration."""
        self.test_cases = [
            # ─────────────────────────────────────────────────────────────
            # Functional Test Cases
            # ─────────────────────────────────────────────────────────────
            TestCase(
                id="func_001",
                name="Basic Pneumonia Detection",
                description="Test detection of typical pneumonia findings",
                category="functional",
                input_data={
                    "question": "Analyze this chest X-ray for pneumonia",
                    "image_findings": "Right lower lobe consolidation with air bronchograms",
                    "use_rag": True,
                },
                expected_outputs={
                    "should_contain": ["pneumonia", "consolidation"],
                    "should_have_sections": ["findings", "impression"],
                    "min_confidence": 0.6,
                    "should_cite_literature": True,
                },
                tags=["pneumonia", "basic"],
            ),
            TestCase(
                id="func_002",
                name="Normal Chest X-Ray",
                description="Test handling of normal X-ray (no pathology)",
                category="functional",
                input_data={
                    "question": "Analyze this chest X-ray",
                    "image_findings": "Clear lung fields bilaterally. Normal cardiac silhouette.",
                    "use_rag": True,
                },
                expected_outputs={
                    "should_contain": ["normal", "unremarkable"],
                    "should_not_contain": ["pneumonia", "mass", "nodule"],
                    "should_have_sections": ["findings", "impression"],
                },
                tags=["normal", "basic"],
            ),
            TestCase(
                id="func_003",
                name="Multiple Findings",
                description="Test handling of multiple concurrent findings",
                category="functional",
                input_data={
                    "question": "Provide comprehensive analysis",
                    "image_findings": "Cardiomegaly with pulmonary edema. Small left pleural effusion.",
                    "use_rag": True,
                },
                expected_outputs={"should_contain": ["cardiomegaly", "edema", "effusion"], "min_findings_count": 2},
                tags=["multi-finding", "complex"],
            ),
            TestCase(
                id="func_004",
                name="RAG vs Non-RAG Comparison",
                description="Verify RAG adds literature citations",
                category="functional",
                input_data={
                    "question": "Analyze for pneumonia",
                    "image_findings": "Lobar consolidation right lower lobe",
                    "use_rag": True,
                },
                expected_outputs={
                    "should_cite_literature": True,
                    "should_contain": ["source", "literature", "guideline"],
                },
                tags=["rag", "comparison"],
            ),
            # ─────────────────────────────────────────────────────────────
            # Performance Test Cases
            # ─────────────────────────────────────────────────────────────
            TestCase(
                id="perf_001",
                name="Latency - Simple Query",
                description="Test response time for simple query",
                category="performance",
                input_data={"question": "Quick analysis", "image_findings": "Normal chest X-ray", "use_rag": False},
                expected_outputs={"max_latency_ms": 5000},
                timeout_seconds=10.0,
                tags=["latency", "simple"],
            ),
            TestCase(
                id="perf_002",
                name="Latency - RAG Query",
                description="Test response time with RAG enabled",
                category="performance",
                input_data={
                    "question": "Detailed analysis with literature",
                    "image_findings": "Right lower lobe opacity",
                    "use_rag": True,
                },
                expected_outputs={"max_latency_ms": 15000},
                timeout_seconds=30.0,
                tags=["latency", "rag"],
            ),
            # ─────────────────────────────────────────────────────────────
            # Robustness Test Cases
            # ─────────────────────────────────────────────────────────────
            TestCase(
                id="robust_001",
                name="Empty Question Handling",
                description="Test handling of empty or minimal question",
                category="robustness",
                input_data={"question": "", "image_findings": "Normal chest X-ray", "use_rag": True},
                expected_outputs={"should_not_error": True, "should_produce_output": True},
                tags=["edge-case", "empty-input"],
            ),
            TestCase(
                id="robust_002",
                name="Very Long Question",
                description="Test handling of excessively long question",
                category="robustness",
                input_data={
                    "question": "Analyze " * 100 + "this chest X-ray for any abnormalities",
                    "image_findings": "Normal chest X-ray",
                    "use_rag": True,
                },
                expected_outputs={"should_not_error": True, "should_produce_output": True},
                tags=["edge-case", "long-input"],
            ),
            TestCase(
                id="robust_003",
                name="Ambiguous Findings",
                description="Test handling of ambiguous/unclear findings",
                category="robustness",
                input_data={
                    "question": "What do you see?",
                    "image_findings": "Subtle opacity, unclear significance",
                    "use_rag": True,
                },
                expected_outputs={
                    "should_indicate_uncertainty": True,
                    "should_contain": ["uncertain", "possible", "may", "could"],
                },
                tags=["edge-case", "ambiguous"],
            ),
        ]

    def load_test_cases(self, path: str) -> None:
        """
        Load test cases from JSON file.

        Expected format:
        [
            {
                "id": "test_001",
                "name": "Test Name",
                "description": "...",
                "category": "functional",
                "input_data": {...},
                "expected_outputs": {...}
            },
            ...
        ]
        """
        with open(path, "r") as f:
            data = json.load(f)

        self.test_cases = [TestCase(**tc) for tc in data]

    def save_test_cases(self, path: str) -> None:
        """Save test cases to JSON file."""
        data = [asdict(tc) for tc in self.test_cases]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _mock_pipeline_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate mock pipeline response for testing without actual models.

        This allows running benchmarks in CI environments without GPU/models.
        """
        findings = input_data.get("image_findings", "")
        use_rag = input_data.get("use_rag", True)

        # Generate mock response based on input
        mock_report = f"""
FINDINGS:
{findings}

IMPRESSION:
Based on the imaging findings described above.
"""

        if "pneumonia" in findings.lower() or "consolidation" in findings.lower():
            mock_report += "Findings are consistent with pneumonia.\n"

        if "normal" in findings.lower():
            mock_report += "No acute cardiopulmonary abnormality. Unremarkable examination.\n"

        if use_rag:
            mock_report += "\nLiterature sources consulted: [Source 1] ACR Guidelines\n"

        return {
            "initial_findings": findings,
            "literature_context": "[Source 1] Medical literature..." if use_rag else "",
            "diagnostic_reasoning": f"Analysis of: {findings}",
            "final_report": mock_report,
            "confidence": 0.75,
            "num_sources": 3 if use_rag else 0,
        }

    def _evaluate_output(
        self, actual: Dict[str, Any], expected: Dict[str, Any], execution_time_ms: float
    ) -> tuple[bool, float, Dict[str, bool], List[str]]:
        """
        Evaluate actual output against expected outputs.

        Returns:
            (passed, score, checks, errors)
        """
        checks = {}
        errors = []
        scores = []

        # Get the main output text
        output_text = (
            actual.get("final_report", "") + actual.get("enhanced_analysis", "") + actual.get("initial_findings", "")
        ).lower()

        # Check: should_contain
        if "should_contain" in expected:
            for term in expected["should_contain"]:
                check_name = f"contains_{term}"
                found = term.lower() in output_text
                checks[check_name] = found
                scores.append(1.0 if found else 0.0)
                if not found:
                    errors.append(f"Missing expected term: '{term}'")

        # Check: should_not_contain
        if "should_not_contain" in expected:
            for term in expected["should_not_contain"]:
                check_name = f"not_contains_{term}"
                found = term.lower() in output_text
                checks[check_name] = not found
                scores.append(1.0 if not found else 0.0)
                if found:
                    errors.append(f"Unexpected term found: '{term}'")

        # Check: should_have_sections
        if "should_have_sections" in expected:
            for section in expected["should_have_sections"]:
                check_name = f"has_section_{section}"
                found = section.lower() in output_text
                checks[check_name] = found
                scores.append(1.0 if found else 0.0)
                if not found:
                    errors.append(f"Missing section: '{section}'")

        # Check: min_confidence
        if "min_confidence" in expected:
            confidence = actual.get("confidence", 0)
            check_name = "min_confidence"
            passed = confidence >= expected["min_confidence"]
            checks[check_name] = passed
            scores.append(1.0 if passed else 0.0)
            if not passed:
                errors.append(f"Confidence {confidence} below threshold {expected['min_confidence']}")

        # Check: should_cite_literature
        if "should_cite_literature" in expected and expected["should_cite_literature"]:
            check_name = "cites_literature"
            has_citation = any(
                marker in output_text for marker in ["source", "[1]", "literature", "guideline", "study"]
            )
            checks[check_name] = has_citation
            scores.append(1.0 if has_citation else 0.0)
            if not has_citation:
                errors.append("No literature citations found")

        # Check: max_latency_ms
        if "max_latency_ms" in expected:
            check_name = "latency"
            passed = execution_time_ms <= expected["max_latency_ms"]
            checks[check_name] = passed
            scores.append(1.0 if passed else 0.0)
            if not passed:
                errors.append(f"Latency {execution_time_ms}ms exceeds max {expected['max_latency_ms']}ms")

        # Check: should_not_error
        if "should_not_error" in expected:
            check_name = "no_error"
            has_error = "error" in actual
            checks[check_name] = not has_error
            scores.append(1.0 if not has_error else 0.0)

        # Check: should_produce_output
        if "should_produce_output" in expected:
            check_name = "has_output"
            has_output = len(output_text.strip()) > 10
            checks[check_name] = has_output
            scores.append(1.0 if has_output else 0.0)
            if not has_output:
                errors.append("No meaningful output produced")

        # Check: should_indicate_uncertainty
        if "should_indicate_uncertainty" in expected:
            check_name = "indicates_uncertainty"
            uncertainty_markers = ["uncertain", "possible", "may", "could", "unclear", "suggest"]
            has_uncertainty = any(m in output_text for m in uncertainty_markers)
            checks[check_name] = has_uncertainty
            scores.append(1.0 if has_uncertainty else 0.0)

        # Calculate overall
        overall_score = sum(scores) / len(scores) if scores else 0.0
        passed = all(checks.values()) if checks else False

        return passed, overall_score, checks, errors

    def run_test(
        self, test_case: TestCase, pipeline_func: Optional[Callable] = None, use_mock: bool = False
    ) -> TestResult:
        """
        Run a single test case.

        Args:
            test_case: The test case to run
            pipeline_func: Optional function to call for actual pipeline execution
            use_mock: If True, use mock responses

        Returns:
            TestResult with execution details
        """
        start_time = time.time()
        errors = []

        try:
            if use_mock or pipeline_func is None:
                actual_output = self._mock_pipeline_response(test_case.input_data)
            else:
                # Call actual pipeline
                actual_output = pipeline_func(
                    image_path=test_case.input_data.get("image_path", ""),
                    question=test_case.input_data.get("question", ""),
                    use_rag=test_case.input_data.get("use_rag", True),
                )

        except Exception as e:
            actual_output = {"error": str(e)}
            errors.append(f"Execution error: {str(e)}")

        execution_time_ms = (time.time() - start_time) * 1000

        # Evaluate output
        passed, score, checks, eval_errors = self._evaluate_output(
            actual_output, test_case.expected_outputs, execution_time_ms
        )
        errors.extend(eval_errors)

        return TestResult(
            test_id=test_case.id,
            test_name=test_case.name,
            passed=passed,
            score=score,
            execution_time_ms=execution_time_ms,
            actual_outputs=actual_output,
            checks=checks,
            errors=errors,
        )

    def run_category(
        self, category: str, pipeline_func: Optional[Callable] = None, use_mock: bool = False
    ) -> BenchmarkResults:
        """
        Run all tests in a category.

        Args:
            category: Category name (functional, performance, robustness)
            pipeline_func: Optional pipeline function
            use_mock: Use mock responses

        Returns:
            BenchmarkResults for the category
        """
        category_tests = [tc for tc in self.test_cases if tc.category == category]
        results = []

        for test_case in category_tests:
            result = self.run_test(test_case, pipeline_func, use_mock)
            results.append(result)

        # Aggregate
        passed_count = sum(1 for r in results if r.passed)
        total = len(results)

        return BenchmarkResults(
            suite_name=f"{category}_benchmarks",
            total_tests=total,
            passed_tests=passed_count,
            failed_tests=total - passed_count,
            pass_rate=passed_count / total if total > 0 else 0.0,
            avg_score=sum(r.score for r in results) / total if total > 0 else 0.0,
            avg_execution_time_ms=sum(r.execution_time_ms for r in results) / total if total > 0 else 0.0,
            test_results=results,
        )

    def run_all(self, pipeline_func: Optional[Callable] = None, use_mock: bool = False) -> Dict[str, BenchmarkResults]:
        """
        Run all benchmark categories.

        Args:
            pipeline_func: Optional pipeline function
            use_mock: Use mock responses

        Returns:
            Dictionary mapping category to BenchmarkResults
        """
        categories = set(tc.category for tc in self.test_cases)
        results = {}

        for category in categories:
            results[category] = self.run_category(category, pipeline_func, use_mock)

        return results

    def generate_report(self, results: Dict[str, BenchmarkResults]) -> str:
        """Generate human-readable benchmark report."""
        lines = ["=" * 70, "END-TO-END BENCHMARK REPORT", "=" * 70, f"Generated: {datetime.now().isoformat()}", ""]

        # Summary
        total_tests = sum(r.total_tests for r in results.values())
        total_passed = sum(r.passed_tests for r in results.values())
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0

        lines.extend(
            [
                "SUMMARY",
                "-" * 70,
                f"Total Tests: {total_tests}",
                f"Passed: {total_passed}",
                f"Failed: {total_tests - total_passed}",
                f"Pass Rate: {overall_pass_rate:.1%}",
                "",
            ]
        )

        # Per-category breakdown
        for category, result in results.items():
            lines.extend(
                [
                    f"\n{category.upper()} BENCHMARKS",
                    "-" * 70,
                    f"Tests: {result.total_tests} | Passed: {result.passed_tests} | "
                    f"Pass Rate: {result.pass_rate:.1%}",
                    f"Avg Score: {result.avg_score:.2f} | Avg Time: {result.avg_execution_time_ms:.0f}ms",
                    "",
                ]
            )

            for test_result in result.test_results:
                status = "PASS" if test_result.passed else "FAIL"
                lines.append(
                    f"  [{status}] {test_result.test_name} "
                    f"(score: {test_result.score:.2f}, time: {test_result.execution_time_ms:.0f}ms)"
                )
                if test_result.errors:
                    for error in test_result.errors[:2]:  # Show first 2 errors
                        lines.append(f"         ! {error}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# =============================================================================
# CLI for standalone benchmarking
# =============================================================================

if __name__ == "__main__":
    print("Running E2E Benchmarks (Mock Mode)...")
    print()

    benchmark = E2EBenchmark()
    results = benchmark.run_all(use_mock=True)

    report = benchmark.generate_report(results)
    print(report)

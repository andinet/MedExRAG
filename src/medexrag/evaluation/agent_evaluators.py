"""
Agent Evaluators for Multi-Agent RAG System

This module provides evaluation functions for each agent in the multi-agent
workflow. Unlike end-to-end evaluation, agent-level evaluation helps identify
which specific component is causing issues.

Agents Evaluated:
    1. Analyst Agent - X-ray finding extraction quality
    2. Researcher Agent - Literature retrieval quality
    3. Diagnostician Agent - Reasoning and citation quality
    4. Reporter Agent - Report completeness and format

Each evaluator can work with:
    - Ground truth comparison (when available)
    - Heuristic checks (always available)
    - LLM-as-judge scoring (optional, requires LLM)

Usage:
    from evaluation.agent_evaluators import AgentEvaluator

    evaluator = AgentEvaluator()

    # Evaluate analyst output
    analyst_score = evaluator.evaluate_analyst(
        findings="Right lower lobe consolidation with air bronchograms",
        ground_truth_findings=["consolidation", "air bronchograms"],
        image_type="chest_xray"
    )

    # Evaluate full agent trace
    trace_eval = evaluator.evaluate_agent_trace(workflow_state)
"""

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentType(Enum):
    """Types of agents in the multi-agent workflow."""

    ANALYST = "analyst"
    RESEARCHER = "researcher"
    DIAGNOSTICIAN = "diagnostician"
    REPORTER = "reporter"


@dataclass
class AgentEvalResult:
    """Evaluation result for a single agent."""

    agent_type: str
    overall_score: float  # 0.0 to 1.0
    component_scores: Dict[str, float]
    issues: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def passed(self, threshold: float = 0.7) -> bool:
        """Check if evaluation passed threshold."""
        return self.overall_score >= threshold


@dataclass
class WorkflowEvalResult:
    """Evaluation result for entire multi-agent workflow."""

    overall_score: float
    agent_scores: Dict[str, AgentEvalResult]
    information_flow_score: float
    latency_score: float
    issues: List[str]
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "agent_scores": {k: v.to_dict() for k, v in self.agent_scores.items()},
            "information_flow_score": self.information_flow_score,
            "latency_score": self.latency_score,
            "issues": self.issues,
            "passed": self.passed,
        }


class AgentEvaluator:
    """
    Evaluator for individual agents and agent interactions.

    This class provides methods to evaluate each agent's output quality
    and the information flow between agents.

    Example:
        evaluator = AgentEvaluator()

        # Evaluate analyst findings
        result = evaluator.evaluate_analyst(
            findings="RLL consolidation, no effusion",
            ground_truth_findings=["consolidation"],
            image_type="chest_xray"
        )

        print(f"Score: {result.overall_score}")
        print(f"Issues: {result.issues}")
    """

    # Medical finding keywords for heuristic evaluation
    CHEST_XRAY_FINDINGS = {
        "consolidation",
        "infiltrate",
        "opacity",
        "effusion",
        "pneumothorax",
        "cardiomegaly",
        "nodule",
        "mass",
        "atelectasis",
        "edema",
        "emphysema",
        "fibrosis",
        "calcification",
        "air bronchogram",
        "silhouette sign",
    }

    ANATOMICAL_LOCATIONS = {
        "right",
        "left",
        "upper",
        "lower",
        "middle",
        "lobe",
        "lung",
        "cardiac",
        "mediastinal",
        "hilar",
        "costophrenic",
        "diaphragm",
        "apex",
        "base",
    }

    REPORT_SECTIONS = {"findings", "impression", "recommendation", "clinical indication", "technique", "comparison"}

    def __init__(self, strict_mode: bool = False):
        """
        Initialize evaluator.

        Args:
            strict_mode: If True, require all checks to pass
        """
        self.strict_mode = strict_mode

    # =========================================================================
    # Analyst Agent Evaluation
    # =========================================================================

    def evaluate_analyst(
        self, findings: str, ground_truth_findings: Optional[List[str]] = None, image_type: str = "chest_xray"
    ) -> AgentEvalResult:
        """
        Evaluate Analyst Agent output.

        Checks:
        1. Finding completeness (if ground truth available)
        2. Anatomical location specificity
        3. Medical terminology usage
        4. Structure and clarity

        Args:
            findings: Text output from analyst agent
            ground_truth_findings: Optional list of expected findings
            image_type: Type of image analyzed

        Returns:
            AgentEvalResult with scores and feedback
        """
        scores = {}
        issues = []
        recommendations = []

        findings_lower = findings.lower()

        # Check 1: Ground truth coverage (if available)
        if ground_truth_findings:
            found_count = sum(1 for gt in ground_truth_findings if gt.lower() in findings_lower)
            coverage = found_count / len(ground_truth_findings)
            scores["ground_truth_coverage"] = coverage

            if coverage < 0.8:
                missing = [gt for gt in ground_truth_findings if gt.lower() not in findings_lower]
                issues.append(f"Missing findings: {missing}")
        else:
            scores["ground_truth_coverage"] = None

        # Check 2: Medical terminology usage
        medical_terms_found = sum(1 for term in self.CHEST_XRAY_FINDINGS if term in findings_lower)
        terminology_score = min(medical_terms_found / 3, 1.0)  # Expect at least 3 terms
        scores["medical_terminology"] = terminology_score

        if medical_terms_found == 0:
            issues.append("No standard medical findings terminology detected")
            recommendations.append("Use standard radiological terms (consolidation, opacity, etc.)")

        # Check 3: Anatomical location specificity
        locations_found = sum(1 for loc in self.ANATOMICAL_LOCATIONS if loc in findings_lower)
        location_score = min(locations_found / 2, 1.0)  # Expect at least 2 locations
        scores["anatomical_specificity"] = location_score

        if locations_found == 0:
            issues.append("No anatomical locations specified")
            recommendations.append("Include specific anatomical locations (e.g., 'right lower lobe')")

        # Check 4: Output length and structure
        word_count = len(findings.split())
        length_score = 1.0 if 20 <= word_count <= 500 else 0.5
        scores["appropriate_length"] = length_score

        if word_count < 20:
            issues.append("Findings too brief")
        elif word_count > 500:
            issues.append("Findings excessively long")

        # Check 5: Negative findings mentioned (good practice)
        negative_patterns = ["no ", "without ", "absent", "unremarkable", "normal"]
        has_negatives = any(p in findings_lower for p in negative_patterns)
        scores["negative_findings"] = 1.0 if has_negatives else 0.5

        if not has_negatives:
            recommendations.append("Consider mentioning pertinent negative findings")

        # Calculate overall score
        valid_scores = [v for v in scores.values() if v is not None]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return AgentEvalResult(
            agent_type=AgentType.ANALYST.value,
            overall_score=overall_score,
            component_scores=scores,
            issues=issues,
            recommendations=recommendations,
            metadata={"word_count": word_count, "image_type": image_type},
        )

    # =========================================================================
    # Researcher Agent Evaluation
    # =========================================================================

    def evaluate_researcher(
        self,
        retrieved_docs: List[Dict[str, Any]],
        query: str,
        expected_sources: Optional[List[str]] = None,
        min_sources: int = 3,
    ) -> AgentEvalResult:
        """
        Evaluate Researcher Agent output.

        Checks:
        1. Number of sources retrieved
        2. Source diversity
        3. Relevance scores (if available)
        4. Source authority (based on metadata)

        Args:
            retrieved_docs: List of retrieved documents with metadata
            query: The search query used
            expected_sources: Optional list of expected source IDs
            min_sources: Minimum expected sources

        Returns:
            AgentEvalResult with scores and feedback
        """
        scores = {}
        issues = []
        recommendations = []

        num_docs = len(retrieved_docs)

        # Check 1: Number of sources
        quantity_score = min(num_docs / min_sources, 1.0)
        scores["source_quantity"] = quantity_score

        if num_docs < min_sources:
            issues.append(f"Only {num_docs} sources retrieved (expected >= {min_sources})")

        # Check 2: Source diversity (unique sources)
        unique_sources = set()
        for doc in retrieved_docs:
            source = doc.get("metadata", {}).get("source", "unknown")
            unique_sources.add(source)

        diversity_score = min(len(unique_sources) / max(num_docs, 1), 1.0)
        scores["source_diversity"] = diversity_score

        if len(unique_sources) == 1 and num_docs > 1:
            issues.append("All results from single source")
            recommendations.append("Consider diversifying literature sources")

        # Check 3: Relevance scores (if available)
        relevance_scores = [doc.get("relevance_score", doc.get("score", 0)) for doc in retrieved_docs]
        if any(r > 0 for r in relevance_scores):
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            scores["avg_relevance"] = avg_relevance

            if avg_relevance < 0.5:
                issues.append(f"Low average relevance score: {avg_relevance:.2f}")
        else:
            scores["avg_relevance"] = None

        # Check 4: Expected sources coverage
        if expected_sources:
            retrieved_sources = {doc.get("metadata", {}).get("source", "") for doc in retrieved_docs}
            found = sum(1 for es in expected_sources if es in retrieved_sources)
            expected_coverage = found / len(expected_sources)
            scores["expected_coverage"] = expected_coverage

            if expected_coverage < 0.5:
                issues.append("Missing expected authoritative sources")
        else:
            scores["expected_coverage"] = None

        # Check 5: Content length (not too short)
        total_content_length = sum(len(doc.get("text", doc.get("content", ""))) for doc in retrieved_docs)
        content_score = 1.0 if total_content_length > 500 else 0.5
        scores["content_richness"] = content_score

        # Calculate overall
        valid_scores = [v for v in scores.values() if v is not None]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return AgentEvalResult(
            agent_type=AgentType.RESEARCHER.value,
            overall_score=overall_score,
            component_scores=scores,
            issues=issues,
            recommendations=recommendations,
            metadata={"num_retrieved": num_docs, "unique_sources": len(unique_sources), "query_length": len(query)},
        )

    # =========================================================================
    # Diagnostician Agent Evaluation
    # =========================================================================

    def evaluate_diagnostician(
        self, diagnosis: str, findings: str, literature: str, expected_diagnoses: Optional[List[str]] = None
    ) -> AgentEvalResult:
        """
        Evaluate Diagnostician Agent output.

        Checks:
        1. Diagnosis presence and clarity
        2. Differential diagnosis included
        3. Evidence citation
        4. Reasoning structure
        5. Confidence indication

        Args:
            diagnosis: Text output from diagnostician agent
            findings: Input findings from analyst
            literature: Input literature from researcher
            expected_diagnoses: Optional list of expected diagnoses

        Returns:
            AgentEvalResult with scores and feedback
        """
        scores = {}
        issues = []
        recommendations = []

        diagnosis_lower = diagnosis.lower()

        # Check 1: Primary diagnosis present
        diagnosis_patterns = [
            r"primary diagnosis[:\s]",
            r"impression[:\s]",
            r"diagnosis[:\s]",
            r"likely\s+\w+",
            r"consistent with",
            r"suggestive of",
        ]
        has_diagnosis = any(re.search(p, diagnosis_lower) for p in diagnosis_patterns)
        scores["diagnosis_present"] = 1.0 if has_diagnosis else 0.0

        if not has_diagnosis:
            issues.append("No clear primary diagnosis statement")
            recommendations.append("Include explicit diagnosis statement")

        # Check 2: Differential diagnosis
        differential_patterns = [r"differential", r"alternative", r"other possibilit", r"could also be", r"less likely"]
        has_differential = any(re.search(p, diagnosis_lower) for p in differential_patterns)
        scores["differential_present"] = 1.0 if has_differential else 0.5

        if not has_differential:
            recommendations.append("Consider including differential diagnoses")

        # Check 3: Citation of evidence
        citation_patterns = [
            r"\[source",
            r"\[\d+\]",
            r"according to",
            r"literature suggests",
            r"guidelines recommend",
            r"studies show",
        ]
        has_citations = any(re.search(p, diagnosis_lower) for p in citation_patterns)
        scores["evidence_cited"] = 1.0 if has_citations else 0.3

        if not has_citations:
            issues.append("No literature citations found")
            recommendations.append("Cite retrieved literature to support diagnosis")

        # Check 4: Uses findings from analyst
        # Check if key terms from findings appear in diagnosis
        finding_terms = set(findings.lower().split()) - {"the", "a", "an", "is", "are", "with"}
        finding_overlap = sum(1 for term in finding_terms if term in diagnosis_lower)
        findings_usage = min(finding_overlap / max(len(finding_terms), 1), 1.0)
        scores["findings_integration"] = findings_usage

        if findings_usage < 0.3:
            issues.append("Diagnosis doesn't reference analyst findings")

        # Check 5: Confidence indication
        confidence_patterns = [
            r"confidence",
            r"likely",
            r"probable",
            r"certain",
            r"\d+%",
            r"high probability",
            r"low probability",
        ]
        has_confidence = any(re.search(p, diagnosis_lower) for p in confidence_patterns)
        scores["confidence_indicated"] = 1.0 if has_confidence else 0.5

        # Check 6: Expected diagnoses coverage
        if expected_diagnoses:
            found = sum(1 for ed in expected_diagnoses if ed.lower() in diagnosis_lower)
            scores["expected_coverage"] = found / len(expected_diagnoses)
        else:
            scores["expected_coverage"] = None

        # Calculate overall
        valid_scores = [v for v in scores.values() if v is not None]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return AgentEvalResult(
            agent_type=AgentType.DIAGNOSTICIAN.value,
            overall_score=overall_score,
            component_scores=scores,
            issues=issues,
            recommendations=recommendations,
            metadata={"diagnosis_length": len(diagnosis)},
        )

    # =========================================================================
    # Reporter Agent Evaluation
    # =========================================================================

    def evaluate_reporter(self, report: str, required_sections: Optional[List[str]] = None) -> AgentEvalResult:
        """
        Evaluate Reporter Agent output.

        Checks:
        1. Required sections present
        2. Professional formatting
        3. Appropriate length
        4. Actionable recommendations

        Args:
            report: Text output from reporter agent
            required_sections: List of required section names

        Returns:
            AgentEvalResult with scores and feedback
        """
        scores = {}
        issues = []
        recommendations = []

        report_lower = report.lower()

        # Default required sections for radiology report
        if required_sections is None:
            required_sections = ["findings", "impression"]

        # Check 1: Required sections present
        sections_found = []
        for section in required_sections:
            if section.lower() in report_lower:
                sections_found.append(section)

        section_score = len(sections_found) / len(required_sections)
        scores["required_sections"] = section_score

        missing_sections = set(required_sections) - set(sections_found)
        if missing_sections:
            issues.append(f"Missing sections: {list(missing_sections)}")

        # Check 2: Professional formatting (headers, structure)
        header_patterns = [
            r"^#+\s",  # Markdown headers
            r"^[A-Z][A-Z\s]+:",  # UPPERCASE HEADERS:
            r"^\*\*\w+",  # Bold headers
        ]
        has_headers = any(re.search(p, report, re.MULTILINE) for p in header_patterns)
        scores["professional_format"] = 1.0 if has_headers else 0.5

        if not has_headers:
            recommendations.append("Consider using clear section headers")

        # Check 3: Appropriate length
        word_count = len(report.split())
        if 100 <= word_count <= 1000:
            length_score = 1.0
        elif 50 <= word_count < 100 or 1000 < word_count <= 1500:
            length_score = 0.7
        else:
            length_score = 0.4
        scores["appropriate_length"] = length_score

        if word_count < 50:
            issues.append("Report too brief")
        elif word_count > 1500:
            issues.append("Report excessively long")

        # Check 4: Recommendations present
        recommendation_patterns = [r"recommend", r"suggest", r"follow.?up", r"consider", r"advised", r"should"]
        has_recommendations = any(re.search(p, report_lower) for p in recommendation_patterns)
        scores["has_recommendations"] = 1.0 if has_recommendations else 0.5

        if not has_recommendations:
            recommendations.append("Include clinical recommendations")

        # Check 5: No placeholder text
        placeholder_patterns = [r"\[.*\]", r"TODO", r"N/A", r"not available"]  # [placeholder]
        has_placeholders = any(re.search(p, report) for p in placeholder_patterns)
        scores["no_placeholders"] = 0.5 if has_placeholders else 1.0

        if has_placeholders:
            issues.append("Report contains placeholder or incomplete text")

        # Calculate overall
        valid_scores = [v for v in scores.values() if v is not None]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return AgentEvalResult(
            agent_type=AgentType.REPORTER.value,
            overall_score=overall_score,
            component_scores=scores,
            issues=issues,
            recommendations=recommendations,
            metadata={"word_count": word_count, "sections_found": sections_found},
        )

    # =========================================================================
    # Workflow-Level Evaluation
    # =========================================================================

    def evaluate_agent_trace(
        self, workflow_state: Dict[str, Any], latency_threshold_ms: float = 10000
    ) -> WorkflowEvalResult:
        """
        Evaluate the entire multi-agent workflow from a state trace.

        Args:
            workflow_state: Final state from LangGraph workflow containing:
                - initial_findings: Analyst output
                - literature_context: Researcher output
                - diagnostic_reasoning: Diagnostician output
                - final_report: Reporter output
                - messages: Message history
            latency_threshold_ms: Maximum acceptable latency

        Returns:
            WorkflowEvalResult with comprehensive evaluation
        """
        agent_scores = {}
        issues = []

        # Evaluate each agent if output is present
        if workflow_state.get("initial_findings"):
            agent_scores["analyst"] = self.evaluate_analyst(findings=workflow_state["initial_findings"])

        if workflow_state.get("literature_context"):
            # Parse literature context if it's a string
            lit_context = workflow_state["literature_context"]
            if isinstance(lit_context, str):
                # Heuristic: count "Source" mentions
                num_sources = lit_context.lower().count("source")
                retrieved_docs = [{"text": lit_context}] if num_sources > 0 else []
            else:
                retrieved_docs = lit_context if isinstance(lit_context, list) else []

            agent_scores["researcher"] = self.evaluate_researcher(
                retrieved_docs=retrieved_docs, query=workflow_state.get("initial_findings", "")
            )

        if workflow_state.get("diagnostic_reasoning"):
            agent_scores["diagnostician"] = self.evaluate_diagnostician(
                diagnosis=workflow_state["diagnostic_reasoning"],
                findings=workflow_state.get("initial_findings", ""),
                literature=workflow_state.get("literature_context", ""),
            )

        if workflow_state.get("final_report"):
            agent_scores["reporter"] = self.evaluate_reporter(report=workflow_state["final_report"])

        # Evaluate information flow
        info_flow_score = self._evaluate_information_flow(workflow_state)

        # Evaluate latency (if timing info available)
        latency_score = 1.0  # Default if no timing
        if "total_latency_ms" in workflow_state:
            latency = workflow_state["total_latency_ms"]
            latency_score = 1.0 if latency <= latency_threshold_ms else 0.5

        # Aggregate issues
        for agent_name, result in agent_scores.items():
            for issue in result.issues:
                issues.append(f"[{agent_name}] {issue}")

        # Calculate overall score
        if agent_scores:
            avg_agent_score = sum(r.overall_score for r in agent_scores.values()) / len(agent_scores)
        else:
            avg_agent_score = 0.0

        overall_score = avg_agent_score * 0.7 + info_flow_score * 0.2 + latency_score * 0.1
        passed = overall_score >= 0.7

        return WorkflowEvalResult(
            overall_score=overall_score,
            agent_scores=agent_scores,
            information_flow_score=info_flow_score,
            latency_score=latency_score,
            issues=issues,
            passed=passed,
        )

    def _evaluate_information_flow(self, state: Dict[str, Any]) -> float:
        """
        Evaluate how well information flows between agents.

        Checks:
        - Analyst findings used in research query
        - Literature used in diagnosis
        - All components appear in final report
        """
        scores = []

        findings = state.get("initial_findings", "")
        literature = state.get("literature_context", "")
        diagnosis = state.get("diagnostic_reasoning", "")
        report = state.get("final_report", "")

        # Check: Findings referenced in diagnosis
        if findings and diagnosis:
            key_terms = [w for w in findings.split()[:10] if len(w) > 4]
            overlap = sum(1 for t in key_terms if t.lower() in diagnosis.lower())
            scores.append(min(overlap / max(len(key_terms), 1), 1.0))

        # Check: Diagnosis appears in report
        if diagnosis and report:
            # Check if key diagnostic terms appear in report
            diag_terms = [w for w in diagnosis.split()[:15] if len(w) > 4]
            overlap = sum(1 for t in diag_terms if t.lower() in report.lower())
            scores.append(min(overlap / max(len(diag_terms), 1), 1.0))

        return sum(scores) / len(scores) if scores else 0.5


# =============================================================================
# CLI for standalone evaluation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Agent Evaluator Demo")
    print("=" * 60)

    evaluator = AgentEvaluator()

    # Demo: Evaluate analyst
    analyst_output = """
    Chest X-ray PA view:

    Findings: There is a right lower lobe consolidation with air bronchograms,
    consistent with pneumonia. No pleural effusion is identified. The cardiac
    silhouette is normal in size. The mediastinum is unremarkable. No pneumothorax.
    """

    print("\n1. Analyst Evaluation:")
    result = evaluator.evaluate_analyst(findings=analyst_output, ground_truth_findings=["consolidation", "pneumonia"])
    print(f"   Score: {result.overall_score:.2f}")
    print(f"   Issues: {result.issues}")

    # Demo: Evaluate diagnostician
    diagnosis_output = """
    DIAGNOSTIC REASONING:

    Primary Diagnosis: Community-acquired pneumonia (high confidence)

    Based on the chest X-ray findings showing right lower lobe consolidation
    with air bronchograms [Source 1], the presentation is most consistent with
    bacterial pneumonia. According to the ACR guidelines [Source 2], this pattern
    is typical for community-acquired pneumonia.

    Differential Diagnosis:
    - Bacterial pneumonia (most likely)
    - Viral pneumonia (less likely given lobar distribution)
    - Atelectasis (no volume loss seen)

    Confidence: 85%
    """

    print("\n2. Diagnostician Evaluation:")
    result = evaluator.evaluate_diagnostician(
        diagnosis=diagnosis_output, findings=analyst_output, literature="ACR guidelines for pneumonia..."
    )
    print(f"   Score: {result.overall_score:.2f}")
    print(f"   Issues: {result.issues}")

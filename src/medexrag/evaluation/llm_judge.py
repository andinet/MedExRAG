"""
LLM-as-Judge Evaluation for Multi-Agent RAG System

This module implements LLM-based evaluation where a language model scores
the quality of outputs. This approach is useful when:
- Ground truth is expensive to collect
- Human-like judgment is needed
- Evaluating subjective qualities (coherence, helpfulness)

Evaluation Dimensions:
    1. Faithfulness - Does output match retrieved evidence?
    2. Relevance - Does output address the question?
    3. Coherence - Is the reasoning logical and clear?
    4. Clinical Appropriateness - Is the advice medically sound?
    5. Citation Accuracy - Are sources correctly referenced?

This module can work with:
    - OpenAI GPT models (via API)
    - Local LLMs (via transformers)
    - Mock mode (for testing without LLM)

Usage:
    from evaluation.llm_judge import LLMJudge

    # With mock (for CI/testing)
    judge = LLMJudge(use_mock=True)
    score = judge.evaluate_faithfulness(output, context)

    # With OpenAI
    judge = LLMJudge(model="gpt-4", api_key="...")
    scores = judge.evaluate_all(output, context, question)

References:
    - G-Eval: https://arxiv.org/abs/2303.16634
    - Prometheus: https://arxiv.org/abs/2310.08491
"""

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class EvaluationDimension(Enum):
    """Dimensions for LLM-based evaluation."""

    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    CLINICAL_APPROPRIATENESS = "clinical_appropriateness"
    CITATION_ACCURACY = "citation_accuracy"
    COMPLETENESS = "completeness"


@dataclass
class JudgeScore:
    """Score from LLM judge for a single dimension."""

    dimension: str
    score: float  # 1-5 scale
    normalized_score: float  # 0-1 scale
    reasoning: str
    confidence: float  # Judge's confidence in the score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeResult:
    """Complete evaluation result from LLM judge."""

    dimension_scores: Dict[str, JudgeScore]
    overall_score: float  # Average normalized score
    overall_reasoning: str
    model_used: str
    is_mock: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension_scores": {k: v.to_dict() for k, v in self.dimension_scores.items()},
            "overall_score": self.overall_score,
            "overall_reasoning": self.overall_reasoning,
            "model_used": self.model_used,
            "is_mock": self.is_mock,
        }


# =============================================================================
# Evaluation Prompts
# =============================================================================

EVALUATION_PROMPTS = {
    EvaluationDimension.FAITHFULNESS: """
You are evaluating the FAITHFULNESS of a medical analysis.
Faithfulness measures whether the output is supported by the provided evidence.

Context (Retrieved Literature):
{context}

Output to Evaluate:
{output}

Score the faithfulness from 1 to 5:
1 - Contains claims completely unsupported by context (hallucination)
2 - Mostly unsupported, significant hallucinations
3 - Mixed - some claims supported, some not
4 - Mostly faithful, minor unsupported details
5 - Completely faithful to the provided context

Respond in JSON format:
{{"score": <1-5>, "reasoning": "<explanation>", "confidence": <0.0-1.0>}}
""",
    EvaluationDimension.RELEVANCE: """
You are evaluating the RELEVANCE of a medical analysis response.
Relevance measures whether the output addresses the user's question.

User Question:
{question}

Output to Evaluate:
{output}

Score the relevance from 1 to 5:
1 - Completely irrelevant, doesn't address the question
2 - Mostly irrelevant, tangential to the question
3 - Partially relevant, addresses some aspects
4 - Mostly relevant, addresses main question
5 - Highly relevant, thoroughly addresses the question

Respond in JSON format:
{{"score": <1-5>, "reasoning": "<explanation>", "confidence": <0.0-1.0>}}
""",
    EvaluationDimension.COHERENCE: """
You are evaluating the COHERENCE of a medical analysis.
Coherence measures logical flow, clarity, and reasoning quality.

Output to Evaluate:
{output}

Score the coherence from 1 to 5:
1 - Incoherent, illogical, contradictory
2 - Poorly organized, hard to follow
3 - Somewhat coherent, some logical gaps
4 - Well organized, minor clarity issues
5 - Excellent coherence, clear logical flow

Respond in JSON format:
{{"score": <1-5>, "reasoning": "<explanation>", "confidence": <0.0-1.0>}}
""",
    EvaluationDimension.CLINICAL_APPROPRIATENESS: """
You are evaluating the CLINICAL APPROPRIATENESS of a medical analysis.
This measures whether the advice and conclusions are medically sound.

Medical Output to Evaluate:
{output}

Score the clinical appropriateness from 1 to 5:
1 - Dangerous or clearly incorrect medical advice
2 - Significant clinical errors or omissions
3 - Acceptable but with notable clinical gaps
4 - Clinically sound with minor improvements possible
5 - Excellent clinical reasoning and appropriate recommendations

Respond in JSON format:
{{"score": <1-5>, "reasoning": "<explanation>", "confidence": <0.0-1.0>}}
""",
    EvaluationDimension.CITATION_ACCURACY: """
You are evaluating the CITATION ACCURACY of a medical analysis.
This measures whether sources are correctly referenced and used.

Retrieved Sources:
{context}

Output with Citations:
{output}

Score the citation accuracy from 1 to 5:
1 - No citations or completely incorrect citations
2 - Poor citation usage, many errors
3 - Some citations, mixed accuracy
4 - Good citation usage, minor errors
5 - Excellent citation accuracy, sources properly attributed

Respond in JSON format:
{{"score": <1-5>, "reasoning": "<explanation>", "confidence": <0.0-1.0>}}
""",
    EvaluationDimension.COMPLETENESS: """
You are evaluating the COMPLETENESS of a medical radiology report.
This measures whether all necessary sections and information are included.

Output to Evaluate:
{output}

Expected sections for a radiology report:
- Findings (descriptive observations)
- Impression (diagnostic conclusions)
- Recommendations (follow-up suggestions)

Score the completeness from 1 to 5:
1 - Missing most required elements
2 - Missing several important elements
3 - Has basic elements but incomplete
4 - Nearly complete, minor omissions
5 - Fully complete with all expected elements

Respond in JSON format:
{{"score": <1-5>, "reasoning": "<explanation>", "confidence": <0.0-1.0>}}
""",
}


class LLMJudge:
    """
    LLM-based evaluator for multi-agent RAG outputs.

    This class uses an LLM to score outputs on multiple dimensions,
    providing human-like quality assessment.

    Example:
        # Mock mode (for testing)
        judge = LLMJudge(use_mock=True)

        # OpenAI mode
        judge = LLMJudge(
            model="gpt-4",
            api_key=os.environ["OPENAI_API_KEY"]
        )

        # Evaluate single dimension
        score = judge.evaluate_faithfulness(
            output="The X-ray shows pneumonia...",
            context="Retrieved literature about pneumonia..."
        )

        # Evaluate all dimensions
        result = judge.evaluate_all(
            output="Full report...",
            context="Literature...",
            question="Analyze for pneumonia"
        )
    """

    def __init__(
        self, model: str = "gpt-4", api_key: Optional[str] = None, use_mock: bool = False, temperature: float = 0.0
    ):
        """
        Initialize LLM Judge.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            use_mock: If True, return mock scores without calling LLM
            temperature: LLM temperature (0 for deterministic)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.use_mock = use_mock
        self.temperature = temperature
        self._client = None

        if not use_mock and self.api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("Warning: openai package not installed. Using mock mode.")
                self.use_mock = True

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        if self.use_mock or self._client is None:
            return self._mock_response(prompt)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert medical AI evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM call failed: {e}")
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing."""
        # Analyze prompt to generate contextual mock score
        prompt_lower = prompt.lower()

        # Base score varies by what we find in the output
        if "pneumonia" in prompt_lower and "consolidation" in prompt_lower:
            score = 4
            reasoning = "Mock: Output mentions key findings (pneumonia, consolidation)"
        elif "error" in prompt_lower or "hallucination" in prompt_lower:
            score = 2
            reasoning = "Mock: Detected potential issues in output"
        else:
            score = 3
            reasoning = "Mock: Average quality detected"

        return json.dumps({"score": score, "reasoning": reasoning, "confidence": 0.7})

    def _parse_response(self, response: str, dimension: str) -> JudgeScore:
        """Parse LLM response into JudgeScore."""
        try:
            # Try to extract JSON from response
            # Handle cases where LLM wraps JSON in markdown
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())
            score = float(data.get("score", 3))
            reasoning = data.get("reasoning", "No reasoning provided")
            confidence = float(data.get("confidence", 0.5))

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback parsing
            score = 3.0
            reasoning = f"Failed to parse response: {response[:100]}"
            confidence = 0.3

        return JudgeScore(
            dimension=dimension,
            score=score,
            normalized_score=(score - 1) / 4,  # Normalize 1-5 to 0-1
            reasoning=reasoning,
            confidence=confidence,
        )

    def evaluate_dimension(
        self, dimension: EvaluationDimension, output: str, context: str = "", question: str = ""
    ) -> JudgeScore:
        """
        Evaluate output on a single dimension.

        Args:
            dimension: Evaluation dimension
            output: Output text to evaluate
            context: Retrieved context (for faithfulness, citation)
            question: Original question (for relevance)

        Returns:
            JudgeScore for the dimension
        """
        prompt_template = EVALUATION_PROMPTS.get(dimension)
        if not prompt_template:
            raise ValueError(f"Unknown dimension: {dimension}")

        prompt = prompt_template.format(
            output=output, context=context or "No context provided", question=question or "No question provided"
        )

        response = self._call_llm(prompt)
        return self._parse_response(response, dimension.value)

    def evaluate_faithfulness(self, output: str, context: str) -> JudgeScore:
        """Evaluate faithfulness to retrieved context."""
        return self.evaluate_dimension(EvaluationDimension.FAITHFULNESS, output=output, context=context)

    def evaluate_relevance(self, output: str, question: str) -> JudgeScore:
        """Evaluate relevance to the question."""
        return self.evaluate_dimension(EvaluationDimension.RELEVANCE, output=output, question=question)

    def evaluate_coherence(self, output: str) -> JudgeScore:
        """Evaluate coherence and logical flow."""
        return self.evaluate_dimension(EvaluationDimension.COHERENCE, output=output)

    def evaluate_clinical_appropriateness(self, output: str) -> JudgeScore:
        """Evaluate clinical/medical appropriateness."""
        return self.evaluate_dimension(EvaluationDimension.CLINICAL_APPROPRIATENESS, output=output)

    def evaluate_citation_accuracy(self, output: str, context: str) -> JudgeScore:
        """Evaluate citation accuracy."""
        return self.evaluate_dimension(EvaluationDimension.CITATION_ACCURACY, output=output, context=context)

    def evaluate_completeness(self, output: str) -> JudgeScore:
        """Evaluate report completeness."""
        return self.evaluate_dimension(EvaluationDimension.COMPLETENESS, output=output)

    def evaluate_all(
        self, output: str, context: str = "", question: str = "", dimensions: Optional[List[EvaluationDimension]] = None
    ) -> JudgeResult:
        """
        Evaluate output on all (or specified) dimensions.

        Args:
            output: Output text to evaluate
            context: Retrieved context
            question: Original question
            dimensions: List of dimensions to evaluate (default: all)

        Returns:
            JudgeResult with all scores
        """
        if dimensions is None:
            dimensions = list(EvaluationDimension)

        dimension_scores = {}

        for dim in dimensions:
            score = self.evaluate_dimension(dim, output, context, question)
            dimension_scores[dim.value] = score

        # Calculate overall score
        if dimension_scores:
            overall_score = sum(s.normalized_score for s in dimension_scores.values()) / len(dimension_scores)

            # Generate overall reasoning
            low_scores = [(k, v) for k, v in dimension_scores.items() if v.normalized_score < 0.5]
            high_scores = [(k, v) for k, v in dimension_scores.items() if v.normalized_score >= 0.75]

            reasoning_parts = []
            if high_scores:
                reasoning_parts.append(f"Strengths: {', '.join(k for k, v in high_scores)}")
            if low_scores:
                reasoning_parts.append(f"Areas for improvement: {', '.join(k for k, v in low_scores)}")

            overall_reasoning = (
                ". ".join(reasoning_parts) if reasoning_parts else "Average performance across dimensions."
            )
        else:
            overall_score = 0.0
            overall_reasoning = "No dimensions evaluated"

        return JudgeResult(
            dimension_scores=dimension_scores,
            overall_score=overall_score,
            overall_reasoning=overall_reasoning,
            model_used=self.model,
            is_mock=self.use_mock,
        )

    def generate_report(self, result: JudgeResult) -> str:
        """Generate human-readable evaluation report."""
        lines = [
            "=" * 60,
            "LLM JUDGE EVALUATION REPORT",
            "=" * 60,
            f"Model: {result.model_used}" + (" (MOCK)" if result.is_mock else ""),
            f"Overall Score: {result.overall_score:.2f} / 1.00",
            f"Overall Assessment: {result.overall_reasoning}",
            "",
            "DIMENSION SCORES:",
            "-" * 60,
        ]

        for dim_name, score in result.dimension_scores.items():
            status = "✓" if score.normalized_score >= 0.6 else "✗"
            lines.append(f"  {status} {dim_name}: {score.score:.1f}/5 " f"(normalized: {score.normalized_score:.2f})")
            lines.append(f"      Reasoning: {score.reasoning[:80]}...")
            lines.append(f"      Confidence: {score.confidence:.2f}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# CLI for standalone evaluation
# =============================================================================

if __name__ == "__main__":
    print("LLM Judge Evaluation Demo (Mock Mode)")
    print()

    judge = LLMJudge(use_mock=True)

    # Sample output to evaluate
    sample_output = """
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

    sample_context = """
    [Source 1] ACR Appropriateness Criteria: Community-acquired pneumonia
    typically presents as lobar consolidation on chest radiography.
    Air bronchograms are a characteristic finding.
    """

    sample_question = "Analyze this chest X-ray for pneumonia"

    # Run full evaluation
    result = judge.evaluate_all(output=sample_output, context=sample_context, question=sample_question)

    # Print report
    report = judge.generate_report(result)
    print(report)

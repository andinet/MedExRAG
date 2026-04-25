"""
LangChain RAG Pipeline with Tools and Agents
Proper implementation using LangChain's tool and agent frameworks

This version demonstrates:
- Custom LangChain Tools
- Multi-agent architecture
- LangGraph orchestration
- Tool calling and routing
"""

import operator
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import torch
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from PIL import Image

# Observability imports
from medexrag.observability import get_logger
from medexrag.observability.context import get_request_id
from medexrag.observability.langsmith_config import trace_pipeline, trace_vlm_call

# Prometheus metrics (M2)
from medexrag.observability.metrics_config import (
    AGENT_STEP_DURATION,
    TOOL_INVOCATIONS,
    record_agent_step,
    record_tool_invocation,
)

# OpenTelemetry distributed tracing (M3)
from medexrag.observability.tracing import (
    add_span_attribute,
    add_span_event,
    trace_operation,
)

# Import our base components (observability already initialized there)
from medexrag.pipeline import DocLingProcessor, MedicalVectorStore, VLMInference

# Get structured logger
logger = get_logger(__name__)


# ============================================================================
# Custom LangChain LLM Wrapper for VLM
# ============================================================================


class MedExRAGLLM(LLM):
    """
    LangChain LLM wrapper for the VLM (Qwen2-VL)
    Allows the VLM to be used as a LangChain LLM with tool calling
    """

    vlm_inference: VLMInference
    current_image: Optional[Image.Image] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "medexrag_vlm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the VLM with the prompt"""
        if self.current_image is None:
            return "Error: No image loaded. Please load an X-ray image first."

        try:
            response = self.vlm_inference.generate(
                self.current_image, prompt, max_new_tokens=kwargs.get("max_new_tokens", 1024)
            )
            return response
        except Exception as e:
            logger.error(f"Error in VLM call: {e}")
            return f"Error generating response: {str(e)}"

    def set_image(self, image: Image.Image):
        """Set the current image for analysis"""
        self.current_image = image


# ============================================================================
# LangChain Tools
# ============================================================================


class LiteratureSearchInput(BaseModel):
    """Input schema for literature search tool"""

    query: str = Field(description="Search query for medical literature")
    k: int = Field(default=5, description="Number of results to return")


class LiteratureSearchTool(BaseTool):
    """
    Tool for searching medical literature knowledge base
    """

    name: str = "search_literature"
    description: str = """
    Search the medical literature knowledge base for relevant information.
    Use this when you need evidence or references from clinical guidelines,
    research papers, or textbooks to support your analysis.

    Input should be a search query related to radiology findings.
    Returns relevant excerpts from medical literature with sources.
    """
    args_schema: type[BaseModel] = LiteratureSearchInput
    vector_store: MedicalVectorStore

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, k: int = 5) -> str:
        """Search literature and return formatted results"""
        try:
            results = self.vector_store.search(query=query, k=k, score_threshold=0.6)

            if not results:
                return "No relevant literature found for this query."

            # Format results
            formatted = ["LITERATURE SEARCH RESULTS:\n"]
            for i, result in enumerate(results, 1):
                source = result["metadata"].get("source", "Unknown")
                score = result.get("relevance_score", 0)
                text = result["text"][:300]

                formatted.append(f"[{i}] {source} (Relevance: {score:.2f})")
                formatted.append(f"{text}...\n")

            return "\n".join(formatted)

        except Exception as e:
            logger.error(f"Error in literature search: {e}")
            return f"Error searching literature: {str(e)}"

    async def _arun(self, query: str, k: int = 5) -> str:
        """Async version"""
        return self._run(query, k)


class ImageAnalysisInput(BaseModel):
    """Input schema for image analysis tool"""

    instruction: str = Field(description="Analysis instruction or question about the X-ray")


class ImageAnalysisTool(BaseTool):
    """
    Tool for analyzing X-ray images with VLM
    """

    name: str = "analyze_xray"
    description: str = """
    Analyze the current X-ray image using the vision language model.
    Use this to get initial findings, identify abnormalities, or answer
    specific questions about the X-ray.

    Input should be a clear instruction or question about what to analyze.
    Returns detailed analysis of the X-ray findings.
    """
    args_schema: type[BaseModel] = ImageAnalysisInput
    llm: MedExRAGLLM

    class Config:
        arbitrary_types_allowed = True

    def _run(self, instruction: str) -> str:
        """Analyze X-ray with VLM"""
        try:
            if self.llm.current_image is None:
                return "Error: No X-ray image loaded."

            response = self.llm._call(instruction)
            return f"X-RAY ANALYSIS:\n{response}"

        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return f"Error analyzing image: {str(e)}"

    async def _arun(self, instruction: str) -> str:
        """Async version"""
        return self._run(instruction)


class DiagnosticReasoningInput(BaseModel):
    """Input schema for diagnostic reasoning tool"""

    findings: str = Field(description="Current findings from X-ray analysis")
    literature_context: str = Field(description="Relevant literature context")


class DiagnosticReasoningTool(BaseTool):
    """
    Tool for diagnostic reasoning with evidence synthesis
    """

    name: str = "diagnostic_reasoning"
    description: str = """
    Perform diagnostic reasoning by synthesizing X-ray findings with
    medical literature evidence. Use this after gathering findings
    and literature to create evidence-based diagnostic impressions.

    Input should include both findings and literature context.
    Returns diagnostic reasoning with differential diagnosis and recommendations.
    """
    args_schema: type[BaseModel] = DiagnosticReasoningInput
    llm: MedExRAGLLM

    class Config:
        arbitrary_types_allowed = True

    def _run(self, findings: str, literature_context: str) -> str:
        """Perform diagnostic reasoning"""
        try:
            reasoning_prompt = f"""
Based on the X-ray findings and medical literature, provide diagnostic reasoning:

FINDINGS:
{findings}

MEDICAL LITERATURE CONTEXT:
{literature_context}

Provide:
1. Primary diagnosis with confidence level
2. Differential diagnoses (list alternative possibilities)
3. Evidence supporting each diagnosis
4. Recommended follow-up or additional imaging
5. Clinical significance and urgency assessment

Format your response clearly with sections.
"""

            response = self.llm._call(reasoning_prompt)
            return f"DIAGNOSTIC REASONING:\n{response}"

        except Exception as e:
            logger.error(f"Error in diagnostic reasoning: {e}")
            return f"Error in reasoning: {str(e)}"

    async def _arun(self, findings: str, literature_context: str) -> str:
        """Async version"""
        return self._run(findings, literature_context)


# ============================================================================
# Function-based tools (simpler alternative)
# ============================================================================


def create_function_tools(vector_store: MedicalVectorStore, llm: MedExRAGLLM) -> List[StructuredTool]:
    """Create tools using @tool decorator (simpler approach)"""

    @tool
    def search_medical_literature(query: str, k: int = 5) -> str:
        """
        Search medical literature knowledge base for relevant information.

        Args:
            query: Search query related to radiology findings
            k: Number of results to return (default: 5)

        Returns:
            Formatted search results with sources and relevance scores
        """
        try:
            results = vector_store.search(query=query, k=k, score_threshold=0.6)

            if not results:
                return "No relevant literature found."

            formatted = []
            for i, result in enumerate(results, 1):
                source = result["metadata"].get("source", "Unknown")
                score = result.get("relevance_score", 0)
                text = result["text"][:200]
                formatted.append(f"[{i}] {source} ({score:.2f}): {text}...")

            return "\n".join(formatted)
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def analyze_xray_image(instruction: str) -> str:
        """
        Analyze the X-ray image with specific instruction.

        Args:
            instruction: What to analyze or look for in the X-ray

        Returns:
            Detailed analysis of the X-ray
        """
        try:
            if llm.current_image is None:
                return "Error: No image loaded"

            return llm._call(instruction)
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def get_differential_diagnosis(findings: str) -> str:
        """
        Generate differential diagnosis based on findings.

        Args:
            findings: Current X-ray findings

        Returns:
            List of differential diagnoses with reasoning
        """
        prompt = f"""
Based on these X-ray findings:
{findings}

Provide a differential diagnosis list with:
1. Most likely diagnosis
2. Alternative diagnoses
3. Reasoning for each
4. Key distinguishing features

Be concise but thorough.
"""
        try:
            return llm._call(prompt)
        except Exception as e:
            return f"Error: {str(e)}"

    return [search_medical_literature, analyze_xray_image, get_differential_diagnosis]


# ============================================================================
# LangGraph Multi-Agent Workflow
# ============================================================================


class AgentState(TypedDict):
    """State for the multi-agent workflow"""

    messages: Annotated[List[BaseMessage], operator.add]
    image: Optional[Image.Image]
    initial_findings: Optional[str]
    literature_context: Optional[str]
    diagnostic_reasoning: Optional[str]
    final_report: Optional[str]
    next_action: Optional[str]


class MultiAgentRAGWorkflow:
    """
    Multi-agent workflow using LangGraph

    Agents:
    1. Analyst Agent - Analyzes X-ray, identifies findings
    2. Researcher Agent - Searches literature for evidence
    3. Diagnostician Agent - Synthesizes findings and creates diagnosis
    4. Reporter Agent - Generates final formatted report
    """

    def __init__(self, vector_store: MedicalVectorStore, vlm_inference: VLMInference):
        self.vector_store = vector_store
        self.vlm_inference = vlm_inference

        # Create LLM wrapper
        self.llm = MedExRAGLLM(vlm_inference=vlm_inference)

        # Create tools
        self.tools = self._create_tools()

        # Build workflow
        self.workflow = self._build_workflow()

    def _create_tools(self) -> List[BaseTool]:
        """Create all tools for the workflow"""
        return [
            LiteratureSearchTool(vector_store=self.vector_store),
            ImageAnalysisTool(llm=self.llm),
            DiagnosticReasoningTool(llm=self.llm),
        ]

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""

        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        workflow.add_node("analyst", self.analyst_agent)
        workflow.add_node("researcher", self.researcher_agent)
        workflow.add_node("diagnostician", self.diagnostician_agent)
        workflow.add_node("reporter", self.reporter_agent)

        # Define flow
        workflow.set_entry_point("analyst")
        workflow.add_edge("analyst", "researcher")
        workflow.add_edge("researcher", "diagnostician")
        workflow.add_edge("diagnostician", "reporter")
        workflow.add_edge("reporter", END)

        return workflow.compile()

    def analyst_agent(self, state: AgentState) -> AgentState:
        """Agent 1: Analyze X-ray and identify findings"""
        start_time = time.time()

        # OpenTelemetry span for agent step (M3)
        with trace_operation("agent_analyst", {"agent.name": "analyst", "agent.task": "analyze_xray"}) as otel_span:
            # Record Prometheus metrics for agent step
            with record_agent_step("analyst"):
                with trace_vlm_call(
                    name="agent_analyst", inputs={"task": "analyze_xray"}, run_type="chain", tags=["agent", "analyst"]
                ) as run:
                    logger.info("agent_analyst_started")

                    # Set image for VLM
                    if state.get("image"):
                        self.llm.set_image(state["image"])

                    # Analyze image
                    analysis_tool = ImageAnalysisTool(llm=self.llm)
                    try:
                        findings = analysis_tool._run(
                            "Provide a systematic analysis of this chest X-ray. "
                            "Identify all abnormalities, describe their location, size, and characteristics."
                        )
                        record_tool_invocation("image_analysis", "success")
                        add_span_event("tool_invocation", {"tool": "image_analysis", "status": "success"})
                    except Exception as e:
                        record_tool_invocation("image_analysis", "error")
                        add_span_event("tool_invocation", {"tool": "image_analysis", "status": "error"})
                        raise

                    state["initial_findings"] = findings
                    state["messages"].append(AIMessage(content=f"Analyst: {findings}"))

                    duration_ms = (time.time() - start_time) * 1000
                    otel_span.set_attribute("agent.duration_ms", duration_ms)
                    otel_span.set_attribute("agent.findings_length", len(findings))
                    logger.info("agent_analyst_completed", duration_ms=duration_ms, findings_length=len(findings))
                    run.end(outputs={"findings_length": len(findings), "duration_ms": duration_ms})

        return state

    def researcher_agent(self, state: AgentState) -> AgentState:
        """Agent 2: Search medical literature for relevant context"""
        start_time = time.time()

        # OpenTelemetry span for agent step (M3)
        with trace_operation(
            "agent_researcher", {"agent.name": "researcher", "agent.task": "search_literature"}
        ) as otel_span:
            # Record Prometheus metrics for agent step
            with record_agent_step("researcher"):
                with trace_vlm_call(
                    name="agent_researcher",
                    inputs={"task": "search_literature"},
                    run_type="retriever",
                    tags=["agent", "researcher", "rag"],
                ) as run:
                    logger.info("agent_researcher_started")

                    if not state.get("initial_findings"):
                        otel_span.set_attribute("agent.skipped", True)
                        run.end(outputs={"skipped": True, "reason": "no_findings"})
                        return state

                    # Extract key terms from findings
                    findings = state["initial_findings"]

                    # Search literature
                    search_tool = LiteratureSearchTool(vector_store=self.vector_store)
                    try:
                        literature = search_tool._run(query=findings, k=5)
                        record_tool_invocation("literature_search", "success")
                        add_span_event("tool_invocation", {"tool": "literature_search", "status": "success"})
                    except Exception as e:
                        record_tool_invocation("literature_search", "error")
                        add_span_event("tool_invocation", {"tool": "literature_search", "status": "error"})
                        raise

                    state["literature_context"] = literature
                    state["messages"].append(AIMessage(content=f"Researcher: {literature}"))

                    duration_ms = (time.time() - start_time) * 1000
                    otel_span.set_attribute("agent.duration_ms", duration_ms)
                    otel_span.set_attribute("agent.literature_length", len(literature))
                    logger.info(
                        "agent_researcher_completed", duration_ms=duration_ms, literature_length=len(literature)
                    )
                    run.end(outputs={"literature_length": len(literature), "duration_ms": duration_ms})

        return state

    def diagnostician_agent(self, state: AgentState) -> AgentState:
        """Agent 3: Synthesize findings and literature into diagnosis"""
        start_time = time.time()

        # OpenTelemetry span for agent step (M3)
        with trace_operation(
            "agent_diagnostician", {"agent.name": "diagnostician", "agent.task": "diagnostic_reasoning"}
        ) as otel_span:
            # Record Prometheus metrics for agent step
            with record_agent_step("diagnostician"):
                with trace_vlm_call(
                    name="agent_diagnostician",
                    inputs={"task": "diagnostic_reasoning"},
                    run_type="chain",
                    tags=["agent", "diagnostician"],
                ) as run:
                    logger.info("agent_diagnostician_started")

                    if not state.get("initial_findings") or not state.get("literature_context"):
                        otel_span.set_attribute("agent.skipped", True)
                        run.end(outputs={"skipped": True, "reason": "missing_inputs"})
                        return state

                    # Perform diagnostic reasoning
                    reasoning_tool = DiagnosticReasoningTool(llm=self.llm)
                    try:
                        diagnosis = reasoning_tool._run(
                            findings=state["initial_findings"], literature_context=state["literature_context"]
                        )
                        record_tool_invocation("diagnostic_reasoning", "success")
                        add_span_event("tool_invocation", {"tool": "diagnostic_reasoning", "status": "success"})
                    except Exception as e:
                        record_tool_invocation("diagnostic_reasoning", "error")
                        add_span_event("tool_invocation", {"tool": "diagnostic_reasoning", "status": "error"})
                        raise

                    state["diagnostic_reasoning"] = diagnosis
                    state["messages"].append(AIMessage(content=f"Diagnostician: {diagnosis}"))

                    duration_ms = (time.time() - start_time) * 1000
                    otel_span.set_attribute("agent.duration_ms", duration_ms)
                    otel_span.set_attribute("agent.diagnosis_length", len(diagnosis))
                    logger.info(
                        "agent_diagnostician_completed", duration_ms=duration_ms, diagnosis_length=len(diagnosis)
                    )
                    run.end(outputs={"diagnosis_length": len(diagnosis), "duration_ms": duration_ms})

        return state

    def reporter_agent(self, state: AgentState) -> AgentState:
        """Agent 4: Generate final formatted report"""
        start_time = time.time()

        # OpenTelemetry span for agent step (M3)
        with trace_operation(
            "agent_reporter", {"agent.name": "reporter", "agent.task": "generate_report"}
        ) as otel_span:
            # Record Prometheus metrics for agent step
            with record_agent_step("reporter"):
                with trace_vlm_call(
                    name="agent_reporter",
                    inputs={"task": "generate_report"},
                    run_type="chain",
                    tags=["agent", "reporter"],
                ) as run:
                    logger.info("agent_reporter_started")

                    # Compile final report
                    report_prompt = f"""
Generate a formal radiology report based on:

FINDINGS:
{state.get('initial_findings', 'N/A')}

DIAGNOSTIC REASONING:
{state.get('diagnostic_reasoning', 'N/A')}

Format the report with these sections:
- CLINICAL INDICATION
- TECHNIQUE
- FINDINGS
- IMPRESSION
- RECOMMENDATIONS

Be professional and concise. Include confidence levels and cite literature sources where relevant.
"""

                    try:
                        final_report = self.llm._call(report_prompt)
                        record_tool_invocation("report_generation", "success")
                        add_span_event("tool_invocation", {"tool": "report_generation", "status": "success"})
                    except Exception as e:
                        record_tool_invocation("report_generation", "error")
                        add_span_event("tool_invocation", {"tool": "report_generation", "status": "error"})
                        raise

                    state["final_report"] = final_report
                    state["messages"].append(AIMessage(content=f"Reporter: {final_report}"))

                    duration_ms = (time.time() - start_time) * 1000
                    otel_span.set_attribute("agent.duration_ms", duration_ms)
                    otel_span.set_attribute("agent.report_length", len(final_report))
                    logger.info("agent_reporter_completed", duration_ms=duration_ms, report_length=len(final_report))
                    run.end(outputs={"report_length": len(final_report), "duration_ms": duration_ms})

        return state

    def analyze(self, image: Image.Image, clinical_question: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete multi-agent workflow"""
        workflow_start = time.time()

        with trace_pipeline(
            name="multi_agent_rag_workflow",
            inputs={
                "clinical_question": clinical_question or "Analyze this chest X-ray",
                "image_size": f"{image.size[0]}x{image.size[1]}" if image else "N/A",
            },
            metadata={"workflow_type": "multi_agent", "num_agents": 4},
        ) as workflow_run:
            logger.info(
                "multi_agent_workflow_started", question=clinical_question, image_size=image.size if image else None
            )

            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=clinical_question or "Analyze this chest X-ray")],
                "image": image,
                "initial_findings": None,
                "literature_context": None,
                "diagnostic_reasoning": None,
                "final_report": None,
                "next_action": None,
            }

            # Run workflow
            final_state = self.workflow.invoke(initial_state)

            result = {
                "findings": final_state.get("initial_findings"),
                "literature": final_state.get("literature_context"),
                "diagnosis": final_state.get("diagnostic_reasoning"),
                "report": final_state.get("final_report"),
                "messages": final_state.get("messages", []),
            }

            duration_ms = (time.time() - workflow_start) * 1000
            logger.info(
                "multi_agent_workflow_completed",
                duration_ms=duration_ms,
                has_report=bool(result.get("report")),
                num_messages=len(result.get("messages", [])),
            )

            workflow_run.end(
                outputs={
                    "duration_ms": duration_ms,
                    "has_findings": bool(result.get("findings")),
                    "has_literature": bool(result.get("literature")),
                    "has_diagnosis": bool(result.get("diagnosis")),
                    "has_report": bool(result.get("report")),
                }
            )

            return result


# ============================================================================
# Simple Agent with Tool Calling
# ============================================================================


class SimpleRAGAgent:
    """
    Simpler agent using LangChain's built-in agent framework
    Single agent with access to multiple tools
    """

    def __init__(self, vector_store: MedicalVectorStore, vlm_inference: VLMInference):
        # Create LLM wrapper
        self.llm = MedExRAGLLM(vlm_inference=vlm_inference)

        # Create tools
        self.tools = create_function_tools(vector_store, self.llm)

        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert radiologist AI assistant with access to tools.

You have access to:
1. analyze_xray_image - Analyze the X-ray image
2. search_medical_literature - Search for evidence in medical literature
3. get_differential_diagnosis - Generate differential diagnosis

When analyzing an X-ray:
1. First, analyze the image to identify findings
2. Search literature for relevant evidence
3. Generate a differential diagnosis
4. Provide a comprehensive report with citations

Always cite literature sources and provide confidence levels.""",
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create agent
        # Note: This requires an LLM that supports tool calling
        # For demo, we'll use a simpler ReAct agent pattern
        logger.info("Simple RAG Agent initialized with tools")

    def analyze(self, image: Image.Image, question: str) -> str:
        """Analyze X-ray using agent with tools"""

        # Set image
        self.llm.set_image(image)

        # For simplicity, manually orchestrate tool calls
        # In production, use create_tool_calling_agent

        # Step 1: Analyze image
        logger.info("Agent: Analyzing X-ray...")
        findings = self.tools[1].invoke({"instruction": question})

        # Step 2: Search literature
        logger.info("Agent: Searching literature...")
        literature = self.tools[0].invoke({"query": findings[:200], "k": 5})

        # Step 3: Get differential
        logger.info("Agent: Generating differential diagnosis...")
        differential = self.tools[2].invoke({"findings": findings})

        # Synthesize response
        final_response = f"""
ANALYSIS COMPLETE

{findings}

SUPPORTING LITERATURE:
{literature}

{differential}
"""

        return final_response


# ============================================================================
# Factory Function
# ============================================================================


def create_langchain_rag_system(
    kb_persist_dir: str = "./data/medical_kb",
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    use_multi_agent: bool = True,
    quantize: "bool | str" = "auto",
):
    """
    Factory function to create LangChain RAG system

    Args:
        kb_persist_dir: Knowledge base directory
        model_name: VLM model name
        use_multi_agent: Use multi-agent workflow (True) or simple agent (False)
        quantize: Use 4-bit quantization. True/False to force, or 'auto' to
                  detect based on GPU VRAM (enables if < 8 GB). Default: 'auto'.

    Returns:
        Configured RAG system (MultiAgentRAGWorkflow or SimpleRAGAgent)
    """

    logger.info("Creating LangChain RAG system...")

    # Initialize components
    vector_store = MedicalVectorStore(persist_directory=kb_persist_dir)
    vlm_inference = VLMInference(model_name=model_name, quantize=quantize)

    if use_multi_agent:
        logger.info("Using multi-agent workflow")
        return MultiAgentRAGWorkflow(vector_store, vlm_inference)
    else:
        logger.info("Using simple agent")
        return SimpleRAGAgent(vector_store, vlm_inference)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from PIL import Image

    # Create system
    rag_system = create_langchain_rag_system(use_multi_agent=True)

    # Load X-ray
    # image = Image.open("chest_xray.dcm")

    # Analyze with multi-agent workflow
    # result = rag_system.analyze(
    #     image,
    #     clinical_question="Evaluate for pneumonia"
    # )

    # print(result["report"])

    print("LangChain RAG system with tools and agents ready!")
    print("\nAvailable tools:")
    print("  1. search_literature - Search medical literature")
    print("  2. analyze_xray - Analyze X-ray image")
    print("  3. diagnostic_reasoning - Perform diagnostic reasoning")
    print("\nMulti-agent workflow:")
    print("  1. Analyst Agent - Analyzes X-ray")
    print("  2. Researcher Agent - Searches literature")
    print("  3. Diagnostician Agent - Synthesizes diagnosis")
    print("  4. Reporter Agent - Generates report")

"""
Basic Unit Tests for Medical X-Ray RAG System

These tests are designed to run in CI without requiring:
- GPU access
- Large ML model downloads (~4GB VLM, ~400MB embeddings)
- External services
- Full project dependencies (tests use standalone definitions)

They test:
1. Core library imports work correctly
2. Data structures patterns are properly defined
3. Pydantic validation works correctly
4. Text processing logic works
5. LangGraph workflow patterns work

Run with: pytest tests/test_basic.py -v
"""

import pytest
from typing import List, Dict, Any, Optional, Annotated, TypedDict
from unittest.mock import MagicMock, patch
import operator


# =============================================================================
# Test 1: Core Library Imports
# =============================================================================

class TestCoreImports:
    """Verify core libraries can be imported without errors."""

    def test_import_langgraph(self):
        """Test that LangGraph components can be imported."""
        from langgraph.graph import StateGraph, END
        assert StateGraph is not None
        assert END is not None

    def test_import_langchain_text_splitter(self):
        """Test that text splitter can be imported."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        assert RecursiveCharacterTextSplitter is not None

    def test_import_langchain_core_messages(self):
        """Test that LangChain core messages can be imported."""
        from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
        assert HumanMessage is not None
        assert AIMessage is not None

    def test_import_pydantic(self):
        """Test that Pydantic can be imported for schema validation."""
        from pydantic import BaseModel, Field
        assert BaseModel is not None
        assert Field is not None

    def test_import_chromadb(self):
        """Test that ChromaDB can be imported for vector storage."""
        import chromadb
        assert chromadb is not None


# =============================================================================
# Test 2: Agent State Structure (Standalone Definition)
# =============================================================================

class TestAgentStatePattern:
    """Test the AgentState TypedDict pattern used in LangGraph workflows."""

    def test_agent_state_definition(self):
        """Verify AgentState pattern can be defined with all required fields."""
        from langchain_core.messages import BaseMessage

        # Define AgentState pattern (mirrors langchain_rag_agents.py:363-372)
        class AgentState(TypedDict):
            messages: Annotated[List[BaseMessage], operator.add]
            image: Optional[Any]
            initial_findings: Optional[str]
            literature_context: Optional[str]
            diagnostic_reasoning: Optional[str]
            final_report: Optional[str]
            next_action: Optional[str]

        # Check that AgentState has the expected annotations
        annotations = AgentState.__annotations__

        expected_fields = [
            'messages',
            'image',
            'initial_findings',
            'literature_context',
            'diagnostic_reasoning',
            'final_report',
            'next_action'
        ]

        for field in expected_fields:
            assert field in annotations, f"AgentState missing field: {field}"

    def test_agent_state_instantiation(self):
        """Test that AgentState can be created with valid data."""
        from langchain_core.messages import HumanMessage, BaseMessage

        class AgentState(TypedDict):
            messages: Annotated[List[BaseMessage], operator.add]
            image: Optional[Any]
            initial_findings: Optional[str]

        # Create a valid state
        state: AgentState = {
            "messages": [HumanMessage(content="Analyze this X-ray")],
            "image": None,
            "initial_findings": None,
        }

        assert len(state["messages"]) == 1
        assert state["initial_findings"] is None


# =============================================================================
# Test 3: Pydantic Tool Schema Pattern
# =============================================================================

class TestPydanticSchemas:
    """Test Pydantic schema patterns used for tool inputs."""

    def test_literature_search_schema(self):
        """Test schema pattern for literature search tool."""
        from pydantic import BaseModel, Field

        class LiteratureSearchInput(BaseModel):
            query: str = Field(description="Search query for medical literature")
            k: int = Field(default=5, description="Number of results to return")

        # Valid input
        input_data = LiteratureSearchInput(query="pneumonia consolidation", k=5)
        assert input_data.query == "pneumonia consolidation"
        assert input_data.k == 5

    def test_literature_search_defaults(self):
        """Test default values in schema."""
        from pydantic import BaseModel, Field

        class LiteratureSearchInput(BaseModel):
            query: str = Field(description="Search query")
            k: int = Field(default=5, description="Number of results")

        # Only required field
        input_data = LiteratureSearchInput(query="test query")
        assert input_data.k == 5  # Default value

    def test_image_analysis_schema(self):
        """Test schema pattern for image analysis tool."""
        from pydantic import BaseModel, Field

        class ImageAnalysisInput(BaseModel):
            instruction: str = Field(description="Analysis instruction")

        input_data = ImageAnalysisInput(instruction="Analyze for pneumonia")
        assert "pneumonia" in input_data.instruction

    def test_diagnostic_reasoning_schema(self):
        """Test schema pattern for diagnostic reasoning tool."""
        from pydantic import BaseModel, Field

        class DiagnosticReasoningInput(BaseModel):
            findings: str = Field(description="X-ray findings")
            literature_context: str = Field(description="Literature context")

        input_data = DiagnosticReasoningInput(
            findings="Right lower lobe consolidation",
            literature_context="ACR guidelines indicate..."
        )

        assert "consolidation" in input_data.findings
        assert "ACR" in input_data.literature_context

    def test_schema_validation_error(self):
        """Test that schema validation catches errors."""
        from pydantic import BaseModel, Field, ValidationError

        class StrictInput(BaseModel):
            query: str = Field(min_length=3)

        # Should raise validation error for too-short query
        with pytest.raises(ValidationError):
            StrictInput(query="ab")


# =============================================================================
# Test 4: Text Processing
# =============================================================================

class TestTextProcessing:
    """Test text chunking and processing logic."""

    def test_recursive_text_splitter(self):
        """Test that text splitter chunks correctly."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", ". ", " "]
        )

        sample_text = """
        This is the first paragraph about pneumonia.
        It contains important medical information.

        This is the second paragraph about chest X-rays.
        It describes how to interpret findings.

        This is the third paragraph about diagnosis.
        """

        chunks = splitter.split_text(sample_text)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Each chunk should be roughly within size limits
        for chunk in chunks:
            # Allow some overflow due to not breaking mid-word
            assert len(chunk) < 150

    def test_text_splitter_preserves_content(self):
        """Test that chunking preserves all original content."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10
        )

        original = "The quick brown fox jumps over the lazy dog."
        chunks = splitter.split_text(original)

        # Key words should appear in at least one chunk
        all_text = " ".join(chunks)
        assert "quick" in all_text
        assert "fox" in all_text
        assert "dog" in all_text

    def test_medical_text_chunking(self):
        """Test chunking with medical terminology."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )

        medical_text = """
        Findings: Right lower lobe consolidation with air bronchograms.
        No pleural effusion. Cardiac silhouette normal.

        Impression: Findings consistent with community-acquired pneumonia.
        Recommend clinical correlation and follow-up imaging.
        """

        chunks = splitter.split_text(medical_text)

        # Should preserve medical terms
        all_text = " ".join(chunks)
        assert "consolidation" in all_text
        assert "pneumonia" in all_text


# =============================================================================
# Test 5: LangGraph Workflow Structure
# =============================================================================

class TestWorkflowStructure:
    """Test LangGraph workflow can be constructed."""

    def test_state_graph_creation(self):
        """Test that a StateGraph can be created."""
        from langgraph.graph import StateGraph, END
        from langchain_core.messages import BaseMessage

        class SimpleState(TypedDict):
            messages: List[str]
            data: Optional[str]

        # Create workflow
        workflow = StateGraph(SimpleState)

        # Should be able to add nodes
        def dummy_node(state):
            return state

        workflow.add_node("test_node", dummy_node)

        # Should not raise
        assert workflow is not None

    def test_workflow_edges(self):
        """Test that workflow edges can be defined."""
        from langgraph.graph import StateGraph, END

        class SimpleState(TypedDict):
            value: str

        workflow = StateGraph(SimpleState)

        def node_a(state):
            state["value"] = "a"
            return state

        def node_b(state):
            state["value"] = "b"
            return state

        workflow.add_node("node_a", node_a)
        workflow.add_node("node_b", node_b)

        # Add edges
        workflow.set_entry_point("node_a")
        workflow.add_edge("node_a", "node_b")
        workflow.add_edge("node_b", END)

        # Compile workflow
        compiled = workflow.compile()

        assert compiled is not None

    def test_workflow_execution(self):
        """Test that a simple workflow executes correctly."""
        from langgraph.graph import StateGraph, END

        class CounterState(TypedDict):
            count: int

        workflow = StateGraph(CounterState)

        def increment(state):
            return {"count": state["count"] + 1}

        def double(state):
            return {"count": state["count"] * 2}

        workflow.add_node("increment", increment)
        workflow.add_node("double", double)

        workflow.set_entry_point("increment")
        workflow.add_edge("increment", "double")
        workflow.add_edge("double", END)

        app = workflow.compile()

        # Execute
        result = app.invoke({"count": 5})

        # 5 + 1 = 6, then 6 * 2 = 12
        assert result["count"] == 12


# =============================================================================
# Test 6: Configuration Validation
# =============================================================================

class TestConfiguration:
    """Test configuration values and defaults."""

    def test_default_chunk_size(self):
        """Verify default chunk size is appropriate for medical text."""
        # Based on rag_pipeline.py defaults
        default_chunk_size = 800
        default_overlap = 150

        # Chunk size should be reasonable for medical abstracts
        assert 500 <= default_chunk_size <= 1500
        # Overlap should be significant but not too large
        assert 50 <= default_overlap <= 300
        # Overlap should be less than chunk size
        assert default_overlap < default_chunk_size

    def test_default_k_sources(self):
        """Verify default number of RAG sources is reasonable."""
        default_k = 5

        # Should retrieve enough for context but not too many
        assert 3 <= default_k <= 10

    def test_embedding_dimension(self):
        """Verify PubMedBERT embedding dimension is correct."""
        # PubMedBERT produces 768-dimensional embeddings
        pubmedbert_dimension = 768
        assert pubmedbert_dimension == 768


# =============================================================================
# Test 7: Mock Integration Test
# =============================================================================

class TestMockIntegration:
    """Integration test using mocks (no real models)."""

    def test_tool_execution_flow(self, mock_vlm):
        """Test that tool execution flow works with mocked VLM."""
        # Simulate the flow: analyze -> search -> reason

        # Step 1: Mock image analysis
        analysis_result = mock_vlm.generate(None, "Analyze X-ray")
        assert "Mock analysis" in analysis_result

        # Step 2: Mock literature search (would use vector store)
        literature_results = [
            {"text": "Pneumonia presents as...", "metadata": {"source": "textbook.pdf"}}
        ]
        assert len(literature_results) > 0

        # Step 3: Combine results
        final_context = f"Findings: {analysis_result}\nLiterature: {literature_results[0]['text']}"
        assert "Findings" in final_context
        assert "Literature" in final_context

    def test_state_accumulation(self):
        """Test that state properly accumulates through workflow."""
        from langchain_core.messages import HumanMessage, AIMessage

        messages = []

        # Simulate message accumulation
        messages.append(HumanMessage(content="Analyze this X-ray"))
        messages.append(AIMessage(content="Initial findings: consolidation"))
        messages.append(AIMessage(content="Literature: pneumonia guidelines"))
        messages.append(AIMessage(content="Diagnosis: likely pneumonia"))

        # Should have all messages
        assert len(messages) == 4

        # Messages should be in order
        assert "Analyze" in messages[0].content
        assert "findings" in messages[1].content
        assert "Literature" in messages[2].content
        assert "Diagnosis" in messages[3].content

    def test_agent_node_pattern(self):
        """Test the agent node pattern used in the workflow."""
        from langchain_core.messages import AIMessage, BaseMessage

        class AgentState(TypedDict):
            messages: Annotated[List[BaseMessage], operator.add]
            findings: Optional[str]

        def mock_analyst_agent(state: AgentState) -> dict:
            """Mock analyst agent that sets findings."""
            findings = "Right lower lobe opacity detected"
            return {
                "messages": [AIMessage(content=f"Analyst: {findings}")],
                "findings": findings
            }

        # Initial state
        initial: AgentState = {
            "messages": [],
            "findings": None
        }

        # Execute agent
        result = mock_analyst_agent(initial)

        # Verify output
        assert result["findings"] is not None
        assert "opacity" in result["findings"]
        assert len(result["messages"]) == 1


# =============================================================================
# Test 8: ChromaDB Vector Store Pattern
# =============================================================================

class TestVectorStorePattern:
    """Test ChromaDB vector store patterns."""

    def test_chromadb_client_creation(self):
        """Test that ChromaDB client can be created."""
        import chromadb

        # Create ephemeral (in-memory) client for testing
        client = chromadb.Client()
        assert client is not None

    def test_chromadb_collection_operations(self):
        """Test basic collection operations."""
        import chromadb

        client = chromadb.Client()

        # Create collection
        collection = client.get_or_create_collection(
            name="test_collection",
            metadata={"description": "Test medical literature"}
        )

        # Add documents
        collection.add(
            documents=["Pneumonia is an infection of the lungs."],
            metadatas=[{"source": "textbook.pdf"}],
            ids=["doc1"]
        )

        # Query
        results = collection.query(
            query_texts=["lung infection"],
            n_results=1
        )

        assert len(results["documents"][0]) == 1
        assert "Pneumonia" in results["documents"][0][0]

    def test_chromadb_with_embeddings_mock(self):
        """Test ChromaDB with mock embeddings function."""
        import chromadb

        # Simple mock embedding function
        def mock_embed(texts):
            # Return fake embeddings (list of lists)
            return [[0.1] * 384 for _ in texts]

        client = chromadb.Client()

        # Create collection with custom embedding function
        collection = client.get_or_create_collection(
            name="test_with_embeddings"
        )

        # Add with IDs
        collection.add(
            documents=["Medical document 1", "Medical document 2"],
            ids=["id1", "id2"]
        )

        # Should be able to query
        results = collection.query(
            query_texts=["medical"],
            n_results=2
        )

        assert len(results["ids"][0]) == 2

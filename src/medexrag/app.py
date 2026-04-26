"""
Streamlit App for MedExRAG - Medical Expert X-ray Analysis
Interactive chat interface with RAG-enhanced X-ray interpretation

Usage:
    streamlit run streamlit_app.py

    or with custom port:
    streamlit run streamlit_app.py --server.port 8501
"""

import io
import json

# Setup observability with console format for Streamlit (easier to read)
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
from PIL import Image

# Initialize observability BEFORE importing RAG pipeline
from medexrag.observability import get_logger, setup_observability
from medexrag.observability.context import generate_request_id, request_context, set_request_attribute

# Health checks (M4)
from medexrag.observability.health import HealthStatus, configure_health_checker, get_health_checker
from medexrag.observability.langsmith_config import trace_pipeline
from medexrag.observability.logging_config import set_correlation_id, set_session_id

# Prometheus metrics imports (M2)
from medexrag.observability.metrics_config import is_metrics_enabled, start_metrics_server

# OpenTelemetry distributed tracing (M3)
from medexrag.observability.tracing import add_span_attribute, get_current_trace_id, trace_operation

os.environ.setdefault("LOG_FORMAT", "console")  # Use console format for dev
setup_observability()
logger = get_logger(__name__)

# Start Prometheus metrics server (M2)
# This runs in a background thread and exposes /metrics on port 8000
metrics_port = int(os.getenv("METRICS_PORT", "8000"))
if os.getenv("PROMETHEUS_ENABLED", "").lower() == "true":
    if start_metrics_server(port=metrics_port):
        logger.info("prometheus_metrics_server_started", port=metrics_port)
    else:
        logger.warning("prometheus_metrics_server_failed", port=metrics_port)

# Import the RAG pipeline (observability already initialized)
from medexrag.pipeline import MedicalRAGPipeline

# Page configuration
st.set_page_config(
    page_title="MedExRAG - Medical Expert X-ray Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.3rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Persistent upload state — survives a browser refresh by writing the
# uploaded image and a path marker to disk.
UPLOADS_DIR = Path("./data/xray_images/uploads")
LAST_UPLOAD_MARKER = Path("./data/xray_images/.last_upload.txt")


def _record_last_upload(path) -> None:
    """Remember the absolute path of the most recently used X-ray."""
    LAST_UPLOAD_MARKER.parent.mkdir(parents=True, exist_ok=True)
    LAST_UPLOAD_MARKER.write_text(str(Path(path).resolve()), encoding="utf-8")


def persist_uploaded_image(image: Image.Image, original_name: str) -> str:
    """Save an uploaded PIL image to disk and remember it as the last upload."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = Path(original_name).stem.replace(" ", "_") or "xray"
    target = UPLOADS_DIR / f"{timestamp}_{safe_stem}.png"
    image.save(target)
    _record_last_upload(target)
    return str(target.resolve())


def restore_last_upload():
    """Return (image, path) for the previously-loaded X-ray, or (None, None)."""
    if not LAST_UPLOAD_MARKER.exists():
        return None, None
    try:
        path = Path(LAST_UPLOAD_MARKER.read_text(encoding="utf-8").strip())
        if not path.exists():
            return None, None
        image = Image.open(path)
        if image.mode != "L":
            image = image.convert("L")
        return image, str(path)
    except Exception as e:
        logger.warning(f"Failed to restore last upload: {e}")
        return None, None


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "current_image" not in st.session_state:
        st.session_state.current_image = None

    if "current_image_path" not in st.session_state:
        st.session_state.current_image_path = None

    # Survive a browser refresh: restore the last uploaded X-ray from disk.
    if st.session_state.current_image is None:
        restored_image, restored_path = restore_last_upload()
        if restored_image is not None:
            st.session_state.current_image = restored_image
            st.session_state.current_image_path = restored_path

    if "kb_stats" not in st.session_state:
        st.session_state.kb_stats = None

    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


@st.cache_resource
def load_pipeline(kb_path: str = "./data/medical_kb"):
    """Load the RAG pipeline (cached)"""
    try:
        pipeline = MedicalRAGPipeline(kb_persist_dir=kb_path)
        return pipeline
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None


def display_chat_message(role: str, content: str, metadata: Optional[Dict] = None):
    """Display a chat message with styling"""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "🧑" if role == "user" else "🤖"

    st.markdown(
        f"""
    <div class="chat-message {css_class}">
        <div style="font-weight: bold; margin-bottom: 0.5rem;">
            {icon} {role.title()}
        </div>
        <div>{content}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Display metadata if available
    if metadata and role == "assistant":
        with st.expander("📊 Analysis Details", expanded=False):
            if "confidence" in metadata:
                confidence = metadata["confidence"]
                if confidence >= 0.8:
                    conf_class = "confidence-high"
                elif confidence >= 0.6:
                    conf_class = "confidence-medium"
                else:
                    conf_class = "confidence-low"

                st.markdown(
                    f"**Confidence:** <span class='{conf_class}'>{confidence:.1%}</span>", unsafe_allow_html=True
                )

            if "sources" in metadata and metadata["sources"]:
                st.markdown("**Sources Used:**")
                for i, source in enumerate(metadata["sources"], 1):
                    st.markdown(
                        f"""
                    <div class="source-box">
                        [{i}] {source}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            if "num_sources" in metadata:
                st.info(f"Retrieved {metadata['num_sources']} relevant literature sources")


def main():
    """Main Streamlit app"""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">🏥 Medical X-Ray RAG Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evidence-Based Radiology with AI</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Knowledge base path
        kb_path = st.text_input(
            "Knowledge Base Path", value="./data/medical_kb", help="Path to the ChromaDB knowledge base"
        )

        # Initialize pipeline
        if not st.session_state.initialized:
            with st.spinner("Loading RAG pipeline..."):
                st.session_state.pipeline = load_pipeline(kb_path)
                if st.session_state.pipeline:
                    st.session_state.kb_stats = st.session_state.pipeline.get_stats()
                    st.session_state.initialized = True
                    st.success("✅ Pipeline loaded!")
                else:
                    st.error("❌ Failed to load pipeline")

        st.divider()

        # RAG toggle
        st.session_state.use_rag = st.checkbox(
            "Use RAG Enhancement", value=st.session_state.use_rag, help="Enable literature context for analysis"
        )

        if st.session_state.use_rag:
            st.session_state.k_sources = st.slider(
                "Number of Sources", min_value=1, max_value=10, value=5, help="Number of literature sources to retrieve"
            )

        st.divider()

        # Knowledge base stats
        st.subheader("📚 Knowledge Base")
        if st.session_state.kb_stats:
            vs_stats = st.session_state.kb_stats.get("vector_store", {})
            total_chunks = vs_stats.get("total_chunks", 0)

            if total_chunks > 0:
                st.success(f"✅ {total_chunks:,} chunks loaded")
            else:
                st.warning("⚠️ Knowledge base is empty")
                st.info("Upload PDFs in the Literature tab")

        st.divider()

        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()

        # Health Status Display (M4)
        st.subheader("🩺 System Health")

        # Configure health checker with pipeline components.
        # Use the underlying `_vlm` field rather than the `vlm` property so
        # that opening the page does not trigger a multi-minute VLM load.
        # Once the user runs an analysis, `_vlm` becomes the loaded instance.
        if st.session_state.pipeline:
            configure_health_checker(
                vlm=getattr(st.session_state.pipeline, "_vlm", None),
                vector_store=getattr(st.session_state.pipeline, "vector_store", None),
            )

        # Get health status
        health_checker = get_health_checker()

        # Show health check button and status
        if st.button("🔄 Check Health", key="health_check_btn"):
            with st.spinner("Checking system health..."):
                st.session_state.health_status = health_checker.full_check(use_cache=False)

        # Display health status
        if "health_status" not in st.session_state:
            st.session_state.health_status = health_checker.full_check(use_cache=True)

        health = st.session_state.health_status

        # Overall status indicator
        status_icons = {
            HealthStatus.HEALTHY: "🟢",
            HealthStatus.DEGRADED: "🟡",
            HealthStatus.UNHEALTHY: "🔴",
            HealthStatus.UNKNOWN: "⚪",
        }

        overall_icon = status_icons.get(health.status, "⚪")
        st.markdown(f"**Overall:** {overall_icon} {health.status.value.title()}")

        # Component health in expandable section
        with st.expander("Component Details", expanded=False):
            for name, comp in health.components.items():
                icon = status_icons.get(comp.status, "⚪")
                st.markdown(f"{icon} **{name.replace('_', ' ').title()}**: {comp.status.value}")
                if comp.message:
                    st.caption(comp.message)

        # Observability links
        st.divider()
        st.subheader("📊 Observability")
        st.markdown("""
        - [Grafana Dashboard](http://localhost:3000) 📈
        - [Prometheus](http://localhost:9090) 📉
        - [Jaeger Traces](http://localhost:16686) 🔍
        """)

    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Chat Analysis", "📁 Upload X-Ray", "📚 Literature Management", "📊 Batch Processing"]
    )

    # Tab 1: Chat Analysis
    with tab1:
        st.header("Chat with RAG System")

        # Display current image
        col1, col2 = st.columns([1, 2])

        with col1:
            if st.session_state.current_image:
                st.image(st.session_state.current_image, caption="Current X-Ray", use_container_width=True)
                st.caption(
                    f"📄 {Path(st.session_state.current_image_path).name if st.session_state.current_image_path else 'Uploaded image'}"
                )
            else:
                st.info("📤 Upload an X-ray image in the 'Upload X-Ray' tab")

        with col2:
            # Chat history display
            chat_container = st.container()

            with chat_container:
                st.subheader("Conversation History")

                if st.session_state.chat_history:
                    for message in st.session_state.chat_history:
                        display_chat_message(message["role"], message["content"], message.get("metadata"))
                else:
                    st.info("👋 Start a conversation by asking a question about the X-ray!")

        # Chat input
        st.divider()

        # Quick question buttons
        st.subheader("Quick Questions")
        col1, col2, col3 = st.columns(3)

        quick_questions = [
            "What abnormalities are present?",
            "Analyze for pneumonia",
            "Provide a comprehensive report",
            "Check for consolidation",
            "Assess cardiac size",
            "Identify any fractures",
        ]

        for i, question in enumerate(quick_questions):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(question, key=f"quick_{i}"):
                    st.session_state.user_input = question
                    st.rerun()

        # Custom question input
        st.subheader("Ask Your Question")

        # Check if Quick Analysis was requested from Upload tab
        if st.session_state.get("quick_analysis_requested", False):
            st.session_state.user_input = st.session_state.get("quick_analysis_question", "")
            st.session_state.quick_analysis_requested = False
            st.info("💡 Quick Analysis question loaded! Click **Analyze** to run the analysis.")

        user_question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What are the findings in this chest X-ray? Is there evidence of pneumonia?",
            key="user_input",
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # Process question
        if analyze_button and user_question:
            if not st.session_state.current_image:
                st.error("❌ Please upload an X-ray image first!")
            elif not st.session_state.pipeline:
                st.error("❌ Pipeline not initialized!")
            else:
                # Generate request ID for tracing
                request_id = str(uuid.uuid4())[:8]
                set_correlation_id(request_id)

                # Add user message to chat
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_question, "timestamp": datetime.now().isoformat()}
                )

                # Analyze with progress and tracing
                with st.spinner(
                    "🔬 Analyzing X-ray..."
                    if not st.session_state.use_rag
                    else "🔬 Analyzing with literature context..."
                ):
                    # OpenTelemetry root span for request (M3)
                    with request_context(request_id=request_id, session_id=st.session_state.session_id):
                        with trace_operation(
                            "streamlit_analysis_request",
                            attributes={
                                "request.id": request_id,
                                "request.session_id": st.session_state.session_id,
                                "request.use_rag": st.session_state.use_rag,
                                "request.question_length": len(user_question),
                                "request.interface": "streamlit",
                            },
                        ) as otel_root_span:
                            # Trace the entire Streamlit analysis request with LangSmith
                            with trace_pipeline(
                                name="streamlit_analysis_request",
                                inputs={
                                    "request_id": request_id,
                                    "question_length": len(user_question),
                                    "use_rag": st.session_state.use_rag,
                                    "k_sources": st.session_state.get("k_sources", 5),
                                },
                                metadata={"interface": "streamlit"},
                            ) as request_trace:
                                try:
                                    logger.info(
                                        "streamlit_analysis_started",
                                        request_id=request_id,
                                        use_rag=st.session_state.use_rag,
                                    )

                                    # Save image temporarily
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                        st.session_state.current_image.save(tmp.name)
                                        tmp_path = tmp.name

                                    # Run analysis
                                    result = st.session_state.pipeline.analyze_xray(
                                        tmp_path,
                                        question=user_question,
                                        use_rag=st.session_state.use_rag,
                                        k_literature=(
                                            st.session_state.get("k_sources", 5) if st.session_state.use_rag else 0
                                        ),
                                    )

                                    # Get response
                                    if st.session_state.use_rag and "enhanced_analysis" in result:
                                        response = result["enhanced_analysis"]
                                        metadata = {
                                            "confidence": result.get("confidence", 0),
                                            "sources": result.get("sources", []),
                                            "num_sources": result.get("num_sources", 0),
                                        }
                                    else:
                                        response = result.get("initial_findings", "No analysis available")
                                        metadata = {}

                                    # Add assistant message to chat
                                    st.session_state.chat_history.append(
                                        {
                                            "role": "assistant",
                                            "content": response,
                                            "metadata": metadata,
                                            "timestamp": datetime.now().isoformat(),
                                        }
                                    )

                                    # Clean up temp file
                                    Path(tmp_path).unlink()

                                    # Update OpenTelemetry span attributes (M3)
                                    otel_root_span.set_attribute("request.status", "success")
                                    otel_root_span.set_attribute("request.confidence", metadata.get("confidence", 0))
                                    otel_root_span.set_attribute("request.num_sources", metadata.get("num_sources", 0))
                                    otel_root_span.set_attribute("request.response_length", len(response))

                                    # Log and trace success
                                    logger.info(
                                        "streamlit_analysis_completed",
                                        request_id=request_id,
                                        confidence=metadata.get("confidence", 0),
                                        num_sources=metadata.get("num_sources", 0),
                                    )
                                    request_trace.end(
                                        outputs={
                                            "status": "success",
                                            "confidence": metadata.get("confidence", 0),
                                            "num_sources": metadata.get("num_sources", 0),
                                            "response_length": len(response),
                                        }
                                    )

                                    st.rerun()

                                except Exception as e:
                                    logger.error("streamlit_analysis_failed", request_id=request_id, error=str(e))
                                    otel_root_span.set_attribute("request.status", "error")
                                    otel_root_span.set_attribute("request.error", str(e))
                                    request_trace.end(error=str(e))

                                    st.error(f"❌ Error during analysis: {e}")
                                    st.session_state.chat_history.append(
                                        {
                                            "role": "assistant",
                                            "content": f"I encountered an error: {str(e)}",
                                            "timestamp": datetime.now().isoformat(),
                                        }
                                    )

    # Tab 2: Upload X-Ray
    with tab2:
        st.header("Upload X-Ray Image")

        upload_method = st.radio("Choose upload method:", ["Upload File", "Use File Path"], horizontal=True)

        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose X-ray image", type=["jpg", "jpeg", "png", "dcm"], help="Upload DICOM, JPG, or PNG image"
            )

            if uploaded_file:
                try:
                    # Load image
                    if uploaded_file.name.endswith(".dcm"):
                        # Handle DICOM
                        import numpy as np
                        import pydicom
                        from pydicom.pixel_data_handlers.util import apply_voi_lut

                        ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
                        pixel_array = ds.pixel_array
                        pixel_array = apply_voi_lut(pixel_array, ds)

                        # Normalize
                        pixel_array = pixel_array - pixel_array.min()
                        pixel_array = pixel_array / pixel_array.max()
                        pixel_array = (pixel_array * 255).astype(np.uint8)

                        image = Image.fromarray(pixel_array)
                    else:
                        image = Image.open(uploaded_file)

                    # Convert to grayscale if needed
                    if image.mode != "L":
                        image = image.convert("L")

                    persisted_path = persist_uploaded_image(image, uploaded_file.name)
                    st.session_state.current_image = image
                    st.session_state.current_image_path = persisted_path

                    col1, col2 = st.columns(2)

                    with col1:
                        st.success("✅ Image loaded successfully!")
                        st.image(image, caption=uploaded_file.name, use_container_width=True)

                    with col2:
                        st.info("**Image Information**")
                        st.write(f"- **Filename:** {uploaded_file.name}")
                        st.write(f"- **Size:** {image.size}")
                        st.write(f"- **Mode:** {image.mode}")

                        if st.button("🔍 Quick Analysis"):
                            st.session_state.quick_analysis_requested = True
                            st.session_state.quick_analysis_question = (
                                "Provide a comprehensive analysis of this chest X-ray"
                            )
                            st.rerun()

                except Exception as e:
                    st.error(f"❌ Error loading image: {e}")

        else:  # Use File Path
            file_path = st.text_input("Enter file path:", placeholder="/path/to/chest_xray.dcm")

            if st.button("Load Image") and file_path:
                try:
                    if not Path(file_path).exists():
                        st.error(f"❌ File not found: {file_path}")
                    else:
                        image = st.session_state.pipeline.load_image(file_path)
                        st.session_state.current_image = image
                        st.session_state.current_image_path = file_path
                        _record_last_upload(file_path)

                        st.success("✅ Image loaded successfully!")
                        st.image(image, caption=Path(file_path).name, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error loading image: {e}")

    # Tab 3: Literature Management
    with tab3:
        st.header("Medical Literature Management")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Upload Literature")

            # Single PDF upload
            st.write("**Upload Single PDF**")
            pdf_file = st.file_uploader("Choose PDF file", type=["pdf"], key="single_pdf")

            save_intermediate = st.toggle(
                "Save parsed output (.md/.json) to medical_literature/", value=False, key="save_intermediate"
            )

            if pdf_file and st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    try:
                        lit_dir = Path("./data/medical_literature")
                        lit_dir.mkdir(parents=True, exist_ok=True)

                        if save_intermediate:
                            # Save PDF to medical_literature/ and process from there
                            pdf_path = lit_dir / pdf_file.name
                            pdf_path.write_bytes(pdf_file.read())
                            document = st.session_state.pipeline.docling.process_pdf(
                                str(pdf_path), save_intermediate=True, output_dir=str(lit_dir)
                            )
                        else:
                            # Original behavior: temp file, process, delete
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(pdf_file.read())
                                pdf_path = Path(tmp.name)
                            document = st.session_state.pipeline.docling.process_pdf(str(pdf_path))
                            pdf_path.unlink()

                        num_chunks = st.session_state.pipeline.vector_store.add_documents([document])

                        # Update stats
                        st.session_state.kb_stats = st.session_state.pipeline.get_stats()

                        if save_intermediate:
                            stem = Path(pdf_file.name).stem
                            st.success(f"✅ Added {num_chunks} chunks from {pdf_file.name}")
                            st.info(
                                f"Saved: {pdf_file.name}, {stem}_parsed.md, {stem}_parsed.json in data/medical_literature/"
                            )
                        else:
                            st.success(f"✅ Added {num_chunks} chunks from {pdf_file.name}")

                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {e}")

            st.divider()

            # Directory ingestion
            st.write("**Ingest from Directory**")
            directory_path = st.text_input(
                "Enter directory path:", value="./data/medical_literature", placeholder="/path/to/pdf/directory"
            )

            if st.button("Ingest Directory"):
                if not Path(directory_path).exists():
                    st.error(f"❌ Directory not found: {directory_path}")
                else:
                    with st.spinner(f"Ingesting PDFs from {directory_path}..."):
                        try:
                            stats = st.session_state.pipeline.ingest_literature(directory_path)

                            # Update stats
                            st.session_state.kb_stats = st.session_state.pipeline.get_stats()

                            st.success(f"✅ Ingestion complete!")
                            st.write(f"- Documents processed: {stats.get('num_documents', 0)}")
                            st.write(f"- Chunks created: {stats.get('num_chunks', 0)}")

                        except Exception as e:
                            st.error(f"❌ Error ingesting directory: {e}")

        with col2:
            st.subheader("🔍 Search Literature")

            search_query = st.text_input("Search query:", placeholder="e.g., pneumonia consolidation findings")

            k_results = st.slider("Number of results:", 1, 10, 5)

            if st.button("Search") and search_query:
                with st.spinner("Searching..."):
                    try:
                        # Use direct similarity search without score threshold
                        raw_results = st.session_state.pipeline.vector_store.vectorstore.similarity_search(
                            query=search_query, k=k_results
                        )

                        if raw_results:
                            st.success(f"Found {len(raw_results)} results")

                            for i, doc in enumerate(raw_results, 1):
                                source = doc.metadata.get("source", "Unknown")
                                with st.expander(f"Result {i} - {source}"):
                                    st.write(f"**Source:** {source}")
                                    st.write(f"**Chunk ID:** {doc.metadata.get('chunk_id', 'N/A')}")
                                    st.write("**Content:**")
                                    st.write(
                                        doc.page_content[:500] + "..."
                                        if len(doc.page_content) > 500
                                        else doc.page_content
                                    )
                        else:
                            st.warning("No results found")

                    except Exception as e:
                        st.error(f"❌ Error searching: {e}")

            st.divider()

            # Knowledge base statistics
            st.subheader("📊 Statistics")

            if st.button("Refresh Stats"):
                st.session_state.kb_stats = st.session_state.pipeline.get_stats()

            if st.session_state.kb_stats:
                vs_stats = st.session_state.kb_stats.get("vector_store", {})

                st.metric("Total Chunks", f"{vs_stats.get('total_chunks', 0):,}")
                st.write(f"**Collection:** {vs_stats.get('collection_name', 'N/A')}")
                st.write(f"**Model:** {st.session_state.kb_stats.get('model', 'N/A')}")

    # Tab 4: Batch Processing
    with tab4:
        st.header("Batch X-Ray Processing")

        st.info("Process multiple X-rays at once with RAG enhancement")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚙️ Configuration")

            input_dir = st.text_input(
                "Input Directory:", value="./xray_images", help="Directory containing X-ray images"
            )

            output_dir = st.text_input(
                "Output Directory:", value="./data/analysis_results", help="Directory to save results"
            )

            batch_use_rag = st.checkbox("Use RAG Enhancement", value=True, key="batch_rag")

        with col2:
            st.subheader("📊 Status")

            if Path(input_dir).exists():
                image_files = (
                    list(Path(input_dir).glob("*.dcm"))
                    + list(Path(input_dir).glob("*.jpg"))
                    + list(Path(input_dir).glob("*.png"))
                )

                st.metric("Images Found", len(image_files))

                if image_files:
                    st.write("**Sample files:**")
                    for f in image_files[:5]:
                        st.write(f"- {f.name}")
                    if len(image_files) > 5:
                        st.write(f"... and {len(image_files) - 5} more")
            else:
                st.warning(f"⚠️ Directory not found: {input_dir}")

        st.divider()

        if st.button("🚀 Start Batch Processing", type="primary"):
            if not Path(input_dir).exists():
                st.error(f"❌ Input directory not found: {input_dir}")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    image_files = (
                        list(Path(input_dir).glob("*.dcm"))
                        + list(Path(input_dir).glob("*.jpg"))
                        + list(Path(input_dir).glob("*.png"))
                    )

                    if not image_files:
                        st.warning("⚠️ No images found in directory")
                    else:
                        status_text.text(f"Processing {len(image_files)} images...")

                        results = st.session_state.pipeline.batch_analyze(
                            input_dir, output_directory=output_dir, use_rag=batch_use_rag
                        )

                        progress_bar.progress(100)

                        # Display results
                        st.success("✅ Batch processing complete!")

                        success = [r for r in results if r["status"] == "success"]
                        errors = [r for r in results if r["status"] == "error"]

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Total", len(results))
                        with col2:
                            st.metric("Success", len(success))
                        with col3:
                            st.metric("Errors", len(errors))

                        if success and batch_use_rag:
                            avg_conf = sum(r.get("confidence", 0) for r in success) / len(success)
                            st.metric("Average Confidence", f"{avg_conf:.1%}")

                        # Show results
                        st.subheader("Results")

                        for result in results:
                            status_icon = "✅" if result["status"] == "success" else "❌"
                            with st.expander(f"{status_icon} {result['file']}"):
                                st.json(result)

                except Exception as e:
                    st.error(f"❌ Error during batch processing: {e}")


if __name__ == "__main__":
    main()

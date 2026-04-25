"""Medical RAG pipeline.

Public API:
    from medexrag.pipeline import (
        MedicalRAGPipeline,
        DocLingProcessor,
        MedicalVectorStore,
        VLMInference,
        RemoteVLMInference,
    )
"""

from medexrag.observability import setup_observability

setup_observability()

from medexrag.pipeline.inference import RemoteVLMInference, VLMInference
from medexrag.pipeline.orchestrator import MedicalRAGPipeline
from medexrag.pipeline.processing import DocLingProcessor
from medexrag.pipeline.vectorstore import MedicalVectorStore

__all__ = [
    "DocLingProcessor",
    "MedicalVectorStore",
    "VLMInference",
    "RemoteVLMInference",
    "MedicalRAGPipeline",
]

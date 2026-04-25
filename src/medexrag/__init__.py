"""
MedExRAG - Medical Expert X-ray RAG Analysis System

Evidence-based radiology interpretation combining Vision Language Models (VLMs)
with Retrieval-Augmented Generation (RAG) for medical X-ray analysis.

Architecture:
    - DocLingProcessor: Extract text/tables from medical PDFs
    - MedicalVectorStore: ChromaDB + PubMedBERT semantic search
    - VLMInference: Vision Language Model for X-ray analysis
    - MedicalRAGPipeline: Two-stage RAG workflow orchestration

Usage:
    from medexrag.pipeline import MedicalRAGPipeline

    pipeline = MedicalRAGPipeline(load_vlm=True)
    result = pipeline.analyze_xray("chest_xray.jpg", use_rag=True)
"""

__version__ = "1.0.0"
__project__ = "MedExRAG"

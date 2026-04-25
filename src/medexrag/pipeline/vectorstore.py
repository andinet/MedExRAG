"""ChromaDB vector store with PubMedBERT embeddings."""

from typing import Any, Dict, List

import torch
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from medexrag.observability import get_logger
from medexrag.observability.metrics_config import (
    record_literature_search,
    record_vector_search,
    update_vector_store_chunks,
)

logger = get_logger(__name__)


class MedicalVectorStore:
    """Vector store for medical literature using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "medical_literature",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    ):
        logger.info("Initializing vector store...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        logger.info(f"Vector store ready at {persist_directory}")

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> int:
        """Chunk and embed `documents`. Returns number of chunks added."""
        logger.info(f"Adding {len(documents)} documents to vector store...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " "],
        )

        all_texts: List[str] = []
        all_metadatas: List[Dict[str, Any]] = []

        for doc in documents:
            chunks = text_splitter.split_text(doc["text"])
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadatas.append(
                    {
                        "source": doc["source"],
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "title": doc["metadata"].get("title", ""),
                        "path": doc["metadata"].get("path", ""),
                    }
                )

        self.vectorstore.add_texts(texts=all_texts, metadatas=all_metadatas)
        logger.info(f"Added {len(all_texts)} chunks to vector store")
        return len(all_texts)

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Semantic search. Set `score_threshold > 0` to filter by relevance."""
        record_literature_search(has_threshold=(score_threshold > 0))

        with record_vector_search():
            if score_threshold > 0:
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    query=query, k=k, score_threshold=score_threshold
                )
                return [
                    {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": score,
                    }
                    for doc, score in results
                ]

            docs = self.vectorstore.similarity_search(query=query, k=k)
            return [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 1.0,
                }
                for doc in docs
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Return collection size; also publishes the chunk-count gauge."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            update_vector_store_chunks(count)
            return {"total_chunks": count, "collection_name": collection.name}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

#!/usr/bin/env python
"""
Test script for literature ingestion
Tests DocLing processing and ChromaDB vector storage
"""

import sys
from pathlib import Path
from medexrag.pipeline import MedicalRAGPipeline

def main():
    print("=" * 70)
    print("MEDICAL LITERATURE INGESTION TEST")
    print("=" * 70)
    print()

    # Check if medical_literature directory exists and has PDFs
    lit_dir = Path("./data/medical_literature")
    if not lit_dir.exists():
        print("ERROR: data/medical_literature directory not found!")
        sys.exit(1)

    pdf_files = list(lit_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  - {pdf.name} ({size_mb:.2f} MB)")
    print()

    if not pdf_files:
        print("WARNING: No PDF files found in data/medical_literature/")
        sys.exit(1)

    # Initialize pipeline (without VLM)
    print("Initializing RAG Pipeline...")
    try:
        pipeline = MedicalRAGPipeline(
            kb_persist_dir="./data/medical_kb",
            load_vlm=False  # Don't load VLM for literature ingestion
        )
        print("[OK] Pipeline initialized successfully")
        print()
    except Exception as e:
        print(f"[ERROR] initializing pipeline: {e}")
        sys.exit(1)

    # Check initial state
    print("Checking initial knowledge base state...")
    initial_stats = pipeline.get_stats()
    initial_chunks = initial_stats.get("vector_store", {}).get("total_chunks", 0)
    print(f"  Current chunks in database: {initial_chunks}")
    print()

    # Ingest literature
    print("-" * 70)
    print("STARTING INGESTION")
    print("-" * 70)
    print()
    print("Processing PDFs with DocLing...")
    print("(This may take a few minutes depending on PDF size and complexity)")
    print()

    try:
        result = pipeline.ingest_literature("./data/medical_literature")

        print()
        print("-" * 70)
        print("INGESTION COMPLETE")
        print("-" * 70)
        print()

        if result.get("status") == "success":
            print("[SUCCESS]")
            print()
            print(f"  Documents processed: {result.get('num_documents', 0)}")
            print(f"  Chunks created: {result.get('num_chunks', 0)}")
            print()

            # Get updated stats
            final_stats = pipeline.get_stats()
            final_chunks = final_stats.get("vector_store", {}).get("total_chunks", 0)
            print(f"  Knowledge base now contains: {final_chunks} total chunks")
            print(f"  New chunks added: {final_chunks - initial_chunks}")

        else:
            print("[WARNING] PARTIAL SUCCESS or ERROR")
            print(f"  Status: {result.get('status')}")
            print(f"  Message: {result.get('message', 'No message')}")

    except Exception as e:
        print()
        print("[ERROR] during ingestion:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test search functionality
    print()
    print("-" * 70)
    print("TESTING SEARCH FUNCTIONALITY")
    print("-" * 70)
    print()

    test_queries = [
        "pneumonia consolidation",
        "chest x-ray interpretation",
        "pulmonary nodule"
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        try:
            results = pipeline.vector_store.search(query, k=3)
            print(f"  Found {len(results)} results")

            if results:
                for i, result in enumerate(results[:2], 1):  # Show top 2
                    score = result.get('relevance_score', 0)
                    source = result['metadata'].get('source', 'Unknown')
                    text_preview = result['text'][:100] + "..."
                    print(f"  [{i}] {source} (score: {score:.3f})")
                    print(f"      {text_preview}")
            print()
        except Exception as e:
            print(f"  [ERROR] searching: {e}")
            print()

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()
    print(f"[OK] PDF files found: {len(pdf_files)}")
    print(f"[OK] Documents processed: {result.get('num_documents', 0)}")
    print(f"[OK] Chunks created: {result.get('num_chunks', 0)}")
    print(f"[OK] Knowledge base ready: {final_chunks} total chunks")
    print()
    print("Your RAG system is ready to use!")
    print()
    print("Next steps:")
    print("  1. Run: streamlit run src/medexrag/app.py")
    print("  2. Go to 'Literature Management' tab to search")
    print("  3. Upload X-rays and use RAG-enhanced analysis")
    print()

if __name__ == "__main__":
    main()

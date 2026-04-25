"""
Example Usage Scripts for MedExRAG Pipeline
Run these examples to get started with the RAG system
"""

# ============================================================================
# Example 1: Basic Setup and Ingestion
# ============================================================================

def example_1_setup_and_ingest():
    """
    Set up the RAG pipeline and ingest medical literature
    """
    print("="*60)
    print("EXAMPLE 1: Setup and Ingest Literature")
    print("="*60)

    from medexrag.pipeline import MedicalRAGPipeline

    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = MedicalRAGPipeline(kb_persist_dir="./data/medical_kb")

    # Ingest literature from PDF directory
    print("\n2. Ingesting literature...")
    # Replace with your actual PDF directory
    literature_dir = "./data/medical_literature"

    # Check if directory exists
    import os
    if not os.path.exists(literature_dir):
        print(f"   Creating example directory: {literature_dir}")
        os.makedirs(literature_dir, exist_ok=True)
        print(f"   ⚠️  Please add PDF files to {literature_dir}")
        print("   Example sources:")
        print("   - ACR Appropriateness Criteria")
        print("   - Fleischner Society Guidelines")
        print("   - Recent research papers on chest X-ray interpretation")
        return

    stats = pipeline.ingest_literature(literature_dir)

    print(f"\n✓ Ingestion complete!")
    print(f"  Documents: {stats.get('num_documents', 0)}")
    print(f"  Chunks: {stats.get('num_chunks', 0)}")


# ============================================================================
# Example 2: Analyze Single X-Ray (Without RAG)
# ============================================================================

def example_2_basic_analysis():
    """
    Analyze a single X-ray without RAG enhancement
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Basic X-Ray Analysis (No RAG)")
    print("="*60)

    from medexrag.pipeline import MedicalRAGPipeline

    pipeline = MedicalRAGPipeline()

    # Analyze without RAG
    xray_path = "sample_chest_xray.dcm"  # Replace with actual path

    print(f"\nAnalyzing: {xray_path}")
    print("Mode: Basic (no literature context)")

    # Check if file exists
    import os
    if not os.path.exists(xray_path):
        print(f"⚠️  File not found: {xray_path}")
        print("Please provide a valid X-ray image path")
        return

    result = pipeline.analyze_xray(
        xray_path,
        question="What abnormalities are present in this chest X-ray?",
        use_rag=False
    )

    print("\n" + "-"*60)
    print("ANALYSIS:")
    print("-"*60)
    print(result["initial_findings"])


# ============================================================================
# Example 3: RAG-Enhanced Analysis
# ============================================================================

def example_3_rag_analysis():
    """
    Analyze X-ray with RAG enhancement from medical literature
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: RAG-Enhanced Analysis")
    print("="*60)

    from medexrag.pipeline import MedicalRAGPipeline

    pipeline = MedicalRAGPipeline()

    # Check knowledge base
    stats = pipeline.get_stats()
    kb_chunks = stats.get('vector_store', {}).get('total_chunks', 0)

    if kb_chunks == 0:
        print("\n⚠️  Knowledge base is empty!")
        print("Please run Example 1 first to ingest literature")
        return

    print(f"\nKnowledge base: {kb_chunks} chunks available")

    xray_path = "sample_chest_xray.dcm"

    import os
    if not os.path.exists(xray_path):
        print(f"⚠️  File not found: {xray_path}")
        return

    print(f"Analyzing: {xray_path}")
    print("Mode: RAG-enhanced (with literature)")

    result = pipeline.analyze_xray(
        xray_path,
        question="Analyze for pneumonia with evidence-based context",
        use_rag=True,
        k_literature=5
    )

    print("\n" + "-"*60)
    print("INITIAL FINDINGS:")
    print("-"*60)
    print(result["initial_findings"][:300] + "...")

    print("\n" + "-"*60)
    print(f"LITERATURE SOURCES ({result['num_sources']} found):")
    print("-"*60)
    for i, source in enumerate(result["sources"], 1):
        print(f"  [{i}] {source}")

    print("\n" + "-"*60)
    print("ENHANCED ANALYSIS:")
    print("-"*60)
    print(result["enhanced_analysis"])

    print(f"\nConfidence: {result['confidence']:.2%}")


# ============================================================================
# Example 4: Search Literature
# ============================================================================

def example_4_search_literature():
    """
    Search the medical literature knowledge base
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Search Literature")
    print("="*60)

    from medexrag.pipeline import MedicalRAGPipeline

    pipeline = MedicalRAGPipeline()

    queries = [
        "pneumonia chest x-ray findings",
        "consolidation air bronchogram",
        "pleural effusion diagnosis"
    ]

    for query in queries:
        print(f"\nSearching: '{query}'")

        results = pipeline.vector_store.search(
            query=query,
            k=3,
            score_threshold=0.6
        )

        print(f"Found {len(results)} results:")

        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "Unknown")
            score = result.get("relevance_score", 0)
            text = result["text"][:150]

            print(f"\n  [{i}] {source} (Relevance: {score:.2f})")
            print(f"      {text}...")


# ============================================================================
# Example 5: Batch Processing
# ============================================================================

def example_5_batch_processing():
    """
    Batch process multiple X-rays
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Processing")
    print("="*60)

    from medexrag.pipeline import MedicalRAGPipeline
    import os

    pipeline = MedicalRAGPipeline()

    xray_dir = "./data/xray_images"
    output_dir = "./data/analysis_results"

    if not os.path.exists(xray_dir):
        print(f"⚠️  Directory not found: {xray_dir}")
        print("Creating example directory...")
        os.makedirs(xray_dir, exist_ok=True)
        print(f"Please add X-ray images to {xray_dir}")
        return

    print(f"Processing images from: {xray_dir}")
    print(f"Saving results to: {output_dir}")

    results = pipeline.batch_analyze(
        xray_dir,
        output_directory=output_dir,
        use_rag=True
    )

    print("\n" + "-"*60)
    print("BATCH SUMMARY:")
    print("-"*60)

    success = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']

    print(f"Total: {len(results)}")
    print(f"Success: {len(success)}")
    print(f"Errors: {len(errors)}")

    if success:
        avg_conf = sum(r.get('confidence', 0) for r in success) / len(success)
        print(f"Average confidence: {avg_conf:.2%}")


# ============================================================================
# Example 6: Using CLI Tool
# ============================================================================

def example_6_cli_usage():
    """
    Show CLI usage examples
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: CLI Tool Usage")
    print("="*60)

    print("\nThe CLI tool provides easy command-line access:")
    print()
    print("1. Ingest literature:")
    print("   python -m medexrag.cli ingest ./data/medical_literature")
    print()
    print("2. Analyze single X-ray:")
    print("   python -m medexrag.cli analyze chest_xray.dcm --rag")
    print()
    print("3. Analyze with custom question:")
    print("   python -m medexrag.cli analyze xray.dcm --rag \\")
    print("     --question 'Analyze for pneumonia'")
    print()
    print("4. Batch process:")
    print("   python -m medexrag.cli batch ./data/xray_images --output ./data/analysis_results --rag")
    print()
    print("5. Search literature:")
    print("   python -m medexrag.cli search 'pneumonia consolidation'")
    print()
    print("6. Get statistics:")
    print("   python -m medexrag.cli stats")


# ============================================================================
# Main Menu
# ============================================================================

def main():
    """
    Interactive example menu
    """
    print("\n" + "="*60)
    print("MEDEXRAG - EXAMPLE SCRIPTS")
    print("="*60)

    examples = {
        "1": ("Setup and Ingest Literature", example_1_setup_and_ingest),
        "2": ("Basic Analysis (No RAG)", example_2_basic_analysis),
        "3": ("RAG-Enhanced Analysis", example_3_rag_analysis),
        "4": ("Search Literature", example_4_search_literature),
        "5": ("Batch Processing", example_5_batch_processing),
        "6": ("CLI Tool Usage", example_6_cli_usage),
    }

    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  q. Quit")

    while True:
        choice = input("\nSelect example (1-7, or 'q' to quit): ").strip()

        if choice.lower() == 'q':
            print("Goodbye!")
            break

        if choice in examples:
            _, func = examples[choice]
            try:
                func()
            except Exception as e:
                print(f"\n✗ Error running example: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Invalid choice. Please select 1-7 or 'q'")


if __name__ == "__main__":
    # Run the menu
    main()

    # Or run specific examples directly:
    # example_1_setup_and_ingest()
    # example_3_rag_analysis()
    # example_4_search_literature()

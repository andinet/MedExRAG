#!/usr/bin/env python3
"""
CLI Tool for Medical X-Ray RAG Analysis (MedExRAG)

Usage:
    # Ingest literature
    python -m medexrag.cli ingest ./medical_literature

    # Analyze single X-ray
    python -m medexrag.cli analyze chest_xray.dcm --question "Analyze for pneumonia"

    # Batch analyze
    python -m medexrag.cli batch ./xray_images --output ./results

    # Search literature
    python -m medexrag.cli search "pneumonia consolidation findings"

    # Get stats
    python -m medexrag.cli stats
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Initialize observability before importing pipeline
from medexrag.observability import get_logger, setup_observability
from medexrag.observability.metrics_config import (
    record_ingestion,
    record_literature_search,
    update_vector_store_chunks,
)

# Setup observability (logging, LangSmith, Prometheus if enabled)
setup_observability()
logger = get_logger(__name__)

from medexrag.pipeline import MedicalRAGPipeline


def ingest_command(args):
    """Ingest literature command"""
    logger.info("cli_ingest_started", directory=args.directory)
    print(f"Ingesting literature from: {args.directory}")

    pipeline = MedicalRAGPipeline(kb_persist_dir=args.kb_dir)

    try:
        stats = pipeline.ingest_literature(args.directory)

        # Record metrics
        num_docs = stats.get("num_documents", 0)
        num_chunks = stats.get("num_chunks", 0)
        for _ in range(num_docs):
            record_ingestion("success")
        update_vector_store_chunks(num_chunks)

        logger.info(
            "cli_ingest_completed", documents=num_docs, chunks=num_chunks, status=stats.get("status", "unknown")
        )

        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Documents processed: {num_docs}")
        print(f"Chunks created: {num_chunks}")
        print(f"Status: {stats.get('status', 'unknown')}")

    except Exception as e:
        record_ingestion("error")
        logger.error("cli_ingest_failed", error=str(e))
        raise


def analyze_command(args):
    """Analyze X-ray command"""
    logger.info("cli_analyze_started", image=args.image, use_rag=args.rag)
    print(f"Analyzing X-ray: {args.image}")

    pipeline = MedicalRAGPipeline(kb_persist_dir=args.kb_dir)

    # Note: Metrics for pipeline.analyze_xray() are recorded inside medexrag.pipeline
    result = pipeline.analyze_xray(args.image, question=args.question, use_rag=args.rag, k_literature=args.k)

    logger.info("cli_analyze_completed", image=args.image, use_rag=args.rag, num_sources=result.get("num_sources", 0))

    # Display results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Image: {result['image_path']}")
    print(f"Question: {result['question']}")

    if args.rag:
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        print(f"Sources used: {result.get('num_sources', 0)}")
        print("\n" + "-" * 60)
        print("ENHANCED ANALYSIS:")
        print("-" * 60)
        print(result.get("enhanced_analysis", "N/A"))

        print("\n" + "-" * 60)
        print("SOURCES:")
        print("-" * 60)
        for i, source in enumerate(result.get("sources", []), 1):
            print(f"  [{i}] {source}")
    else:
        print("\n" + "-" * 60)
        print("ANALYSIS:")
        print("-" * 60)
        print(result.get("initial_findings", "N/A"))

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n✓ Results saved to: {output_path}")


def batch_command(args):
    """Batch analyze command"""
    print(f"Batch analyzing images from: {args.directory}")

    pipeline = MedicalRAGPipeline(kb_persist_dir=args.kb_dir)

    results = pipeline.batch_analyze(args.directory, output_directory=args.output, use_rag=args.rag)

    # Display summary
    print("\n" + "=" * 60)
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 60)

    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"Total images: {len(results)}")
    print(f"Successful: {len(success)}")
    print(f"Errors: {len(errors)}")

    if success:
        avg_confidence = sum(r.get("confidence", 0) for r in success) / len(success)
        print(f"Average confidence: {avg_confidence:.2%}")

    print(f"\n✓ Results saved to: {args.output}")


def search_command(args):
    """Search literature command"""
    logger.info("cli_search_started", query=args.query, k=args.k)
    print(f"Searching literature for: {args.query}")

    pipeline = MedicalRAGPipeline(kb_persist_dir=args.kb_dir)

    # Note: Vector search metrics are recorded inside the search method
    results = pipeline.vector_store.search(query=args.query, k=args.k, score_threshold=args.threshold)

    # Record search with threshold indicator
    has_threshold = args.threshold is not None and args.threshold > 0
    record_literature_search(has_threshold=has_threshold)

    logger.info("cli_search_completed", query=args.query, results_count=len(results))

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Results found: {len(results)}")

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Source: {result['metadata'].get('source', 'Unknown')}")
        print(f"    Relevance: {result.get('relevance_score', 0):.2f}")
        print(f"    Preview: {result['text'][:200]}...")


def stats_command(args):
    """Get statistics command"""
    logger.info("cli_stats_started", kb_dir=args.kb_dir)

    pipeline = MedicalRAGPipeline(kb_persist_dir=args.kb_dir)
    stats = pipeline.get_stats()

    # Update vector store chunks gauge with current count
    vs_stats = stats.get("vector_store", {})
    update_vector_store_chunks(vs_stats.get("total_chunks", 0))

    logger.info("cli_stats_completed", chunks=vs_stats.get("total_chunks", 0))

    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE STATISTICS")
    print("=" * 60)

    vs_stats = stats.get("vector_store", {})
    print(f"Total chunks: {vs_stats.get('total_chunks', 0)}")
    print(f"Collection: {vs_stats.get('collection_name', 'N/A')}")
    print(f"Model: {stats.get('model', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Medical X-Ray RAG Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest literature
  %(prog)s ingest ./data/medical_literature

  # Analyze with RAG
  %(prog)s analyze chest_xray.dcm --rag --question "Analyze for pneumonia"

  # Batch process
  %(prog)s batch ./xray_images --output ./data/analysis_results --rag

  # Search knowledge base
  %(prog)s search "consolidation pneumonia chest x-ray"
        """,
    )

    parser.add_argument(
        "--kb-dir", default="./data/medical_kb", help="Knowledge base directory (default: ./data/medical_kb)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest medical literature")
    ingest_parser.add_argument("directory", help="Directory containing PDF files")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze X-ray image")
    analyze_parser.add_argument("image", help="Path to X-ray image")
    analyze_parser.add_argument(
        "--question", default="Provide a comprehensive analysis of this chest X-ray", help="Analysis question"
    )
    analyze_parser.add_argument("--rag", action="store_true", help="Use RAG enhancement")
    analyze_parser.add_argument(
        "--k", type=int, default=5, help="Number of literature sources to retrieve (default: 5)"
    )
    analyze_parser.add_argument("--output", help="Save results to file (JSON)")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch analyze X-rays")
    batch_parser.add_argument("directory", help="Directory containing X-ray images")
    batch_parser.add_argument(
        "--output", default="./data/analysis_results", help="Output directory (default: ./data/analysis_results)"
    )
    batch_parser.add_argument("--rag", action="store_true", help="Use RAG enhancement")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search literature")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--k", type=int, default=5, help="Number of results (default: 5)")
    search_parser.add_argument("--threshold", type=float, default=0.6, help="Relevance score threshold (default: 0.6)")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get knowledge base statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == "ingest":
            ingest_command(args)
        elif args.command == "analyze":
            analyze_command(args)
        elif args.command == "batch":
            batch_command(args)
        elif args.command == "search":
            search_command(args)
        elif args.command == "stats":
            stats_command(args)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

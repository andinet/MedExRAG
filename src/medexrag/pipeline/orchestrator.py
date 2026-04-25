"""Top-level orchestrator: ingestion + two-stage RAG analysis."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from medexrag.observability import get_logger
from medexrag.observability.langsmith_config import trace_pipeline, trace_vlm_call
from medexrag.observability.metrics_config import (
    record_literature_retrieval,
    record_pipeline_request,
    update_sources_retrieved,
)
from medexrag.observability.tracing import trace_operation
from medexrag.pipeline.inference import RemoteVLMInference, VLMInference
from medexrag.pipeline.processing import DocLingProcessor
from medexrag.pipeline.vectorstore import MedicalVectorStore

logger = get_logger(__name__)


class MedicalRAGPipeline:
    """End-to-end RAG pipeline for medical X-ray analysis."""

    def __init__(
        self,
        kb_persist_dir: str = "./data/medical_kb",
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        load_vlm: bool = False,
        quantize: Union[bool, str] = "auto",
        vlm_endpoint: Optional[str] = None,
    ):
        """
        Args:
            kb_persist_dir: ChromaDB persistence directory.
            model_name: Hugging Face VLM identifier.
            load_vlm: Eagerly load the VLM at construction time.
            quantize: True/False/"auto" — auto enables 4-bit on GPUs <8 GB VRAM.
            vlm_endpoint: If set (or `VLM_ENDPOINT` env var is set), use a remote
                VLM server instead of loading locally.
        """
        logger.info("Initializing Medical RAG Pipeline...")

        self.docling = DocLingProcessor()
        self.vector_store = MedicalVectorStore(persist_directory=kb_persist_dir)

        self.model_name = model_name
        self._quantize = quantize
        self._vlm_endpoint = vlm_endpoint or os.environ.get("VLM_ENDPOINT")
        self._vlm = None
        if load_vlm:
            if self._vlm_endpoint:
                self._vlm = RemoteVLMInference(endpoint=self._vlm_endpoint)
            else:
                self._vlm = VLMInference(model_name=model_name, quantize=quantize)

        logger.info("RAG Pipeline ready!")

    @property
    def vlm(self):
        """Lazy-load the VLM on first access."""
        if self._vlm is None:
            if self._vlm_endpoint:
                logger.info(f"Connecting to remote VLM at: {self._vlm_endpoint}")
                self._vlm = RemoteVLMInference(endpoint=self._vlm_endpoint)
            else:
                logger.info(f"Loading VLM model locally: {self.model_name}")
                try:
                    self._vlm = VLMInference(model_name=self.model_name, quantize=self._quantize)
                except Exception as e:
                    logger.error(f"Error loading VLM: {e}")
                    logger.warning("VLM functionality will be limited without model")
                    raise RuntimeError(
                        f"Failed to load VLM model '{self.model_name}'. "
                        "The model may not be downloaded yet or there may be "
                        f"compatibility issues. Error: {e}"
                    )
        return self._vlm

    def ingest_literature(self, pdf_directory: str) -> Dict[str, Any]:
        """Process every PDF in `pdf_directory` and add to the vector store."""
        logger.info(f"Ingesting literature from {pdf_directory}")

        documents = self.docling.process_directory(pdf_directory)
        if not documents:
            logger.warning("No documents processed!")
            return {"status": "error", "message": "No PDFs processed"}

        num_chunks = self.vector_store.add_documents(documents)
        stats = {
            "status": "success",
            "num_documents": len(documents),
            "num_chunks": num_chunks,
        }
        logger.info(f"Ingestion complete: {stats}")
        return stats

    def load_image(self, image_path: str) -> Image.Image:
        """Load DICOM/JPG/PNG and return a grayscale PIL image."""
        if image_path.endswith(".dcm"):
            import numpy as np
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut

            ds = pydicom.dcmread(image_path)
            pixel_array = ds.pixel_array
            pixel_array = apply_voi_lut(pixel_array, ds)

            pixel_array = pixel_array - pixel_array.min()
            pixel_array = pixel_array / pixel_array.max()
            pixel_array = (pixel_array * 255).astype(np.uint8)

            image = Image.fromarray(pixel_array)
        else:
            image = Image.open(image_path)

        if image.mode != "L":
            image = image.convert("L")

        return image

    def analyze_xray(
        self,
        image_path: str,
        question: str = "Provide a comprehensive analysis of this chest X-ray",
        use_rag: bool = True,
        k_literature: int = 5,
    ) -> Dict[str, Any]:
        """Two-stage analysis: initial findings → literature retrieval → grounded report.

        When `use_rag=False`, only the initial findings are returned.
        """
        pipeline_start = time.time()

        with record_pipeline_request(use_rag=use_rag):
            with trace_operation(
                "analyze_xray_pipeline",
                attributes={
                    "pipeline.image_path": image_path,
                    "pipeline.use_rag": use_rag,
                    "pipeline.k_literature": k_literature,
                    "pipeline.version": "1.0",
                },
            ) as pipeline_span:
                with trace_pipeline(
                    name="analyze_xray_pipeline",
                    inputs={
                        "image_path": image_path,
                        "question": question[:100],
                        "use_rag": use_rag,
                        "k_literature": k_literature,
                    },
                    metadata={"pipeline_version": "1.0"},
                ) as pipeline_run:
                    try:
                        logger.info(
                            "analyze_xray_started",
                            image_path=image_path,
                            use_rag=use_rag,
                            k_literature=k_literature,
                        )

                        with trace_operation("load_image", {"image.path": image_path}) as load_span:
                            image = self.load_image(image_path)
                            load_span.set_attribute("image.size", f"{image.size[0]}x{image.size[1]}")

                        with trace_operation("initial_vlm_analysis", {"analysis.type": "initial"}) as init_span:
                            logger.info("step_initial_analysis", status="started")
                            initial_prompt = (
                                f"{question}\n\nProvide systematic findings " "focusing on key abnormalities."
                            )
                            initial_findings = self.vlm.generate(image, initial_prompt, max_new_tokens=512)
                            init_span.set_attribute("analysis.findings_length", len(initial_findings))
                            logger.info(
                                "step_initial_analysis",
                                status="completed",
                                findings_length=len(initial_findings),
                            )

                        result = {
                            "image_path": image_path,
                            "question": question,
                            "initial_findings": initial_findings,
                        }

                        if not use_rag:
                            duration_ms = (time.time() - pipeline_start) * 1000
                            pipeline_span.set_attribute("pipeline.duration_ms", duration_ms)
                            logger.info(
                                "analyze_xray_completed",
                                use_rag=False,
                                duration_ms=duration_ms,
                            )
                            pipeline_run.end(
                                outputs={
                                    "use_rag": False,
                                    "duration_ms": duration_ms,
                                }
                            )
                            return result

                        with trace_operation(
                            "literature_retrieval",
                            {
                                "retrieval.k": k_literature,
                                "retrieval.query_length": len(initial_findings),
                            },
                        ) as retrieval_span:
                            logger.info(
                                "step_literature_retrieval",
                                status="started",
                                k=k_literature,
                            )
                            retrieval_start = time.time()

                            with record_literature_retrieval():
                                with trace_vlm_call(
                                    name="literature_retrieval",
                                    inputs={
                                        "query_length": len(initial_findings),
                                        "k": k_literature,
                                    },
                                    run_type="retriever",
                                    tags=["rag", "vector-search"],
                                ) as retrieval_run:
                                    literature = self.vector_store.search(query=initial_findings, k=k_literature)
                                    retrieval_duration = (time.time() - retrieval_start) * 1000
                                    retrieval_run.end(
                                        outputs={
                                            "num_results": len(literature),
                                            "duration_ms": retrieval_duration,
                                        }
                                    )

                            retrieval_span.set_attribute("retrieval.num_results", len(literature))
                            retrieval_span.set_attribute("retrieval.duration_ms", retrieval_duration)

                        update_sources_retrieved(len(literature))

                        logger.info(
                            "step_literature_retrieval",
                            status="completed",
                            num_results=len(literature),
                            duration_ms=retrieval_duration,
                        )

                        if not literature:
                            logger.warning("no_relevant_literature_found")
                            result["literature_context"] = []
                            result["enhanced_analysis"] = initial_findings
                            pipeline_span.set_attribute("pipeline.sources_found", 0)
                            pipeline_run.end(outputs={"use_rag": True, "sources_found": 0})
                            return result

                        literature_context = self._format_literature(literature)

                        with trace_operation(
                            "enhanced_vlm_analysis",
                            {
                                "analysis.type": "enhanced",
                                "analysis.num_sources": len(literature),
                            },
                        ) as enhanced_span:
                            logger.info(
                                "step_enhanced_analysis",
                                status="started",
                                num_sources=len(literature),
                            )
                            enhanced_prompt = f"""Based on the X-ray findings and medical literature:

INITIAL FINDINGS:
{initial_findings}

RELEVANT MEDICAL LITERATURE:
{literature_context}

TASK: Provide an evidence-based analysis that:
1. Confirms or refutes initial findings using the literature
2. Adds clinical context from the literature
3. Discusses differential diagnoses
4. Provides evidence-based recommendations
5. Cites specific sources using [Source N] notation

ENHANCED ANALYSIS:"""

                            enhanced_analysis = self.vlm.generate(image, enhanced_prompt, max_new_tokens=1024)
                            enhanced_span.set_attribute("analysis.output_length", len(enhanced_analysis))
                            logger.info(
                                "step_enhanced_analysis",
                                status="completed",
                                analysis_length=len(enhanced_analysis),
                            )

                        confidence = self._calculate_confidence(literature)

                        result.update(
                            {
                                "literature_context": literature,
                                "enhanced_analysis": enhanced_analysis,
                                "confidence": confidence,
                                "num_sources": len(literature),
                                "sources": [lit["metadata"].get("source", "Unknown") for lit in literature],
                            }
                        )

                        duration_ms = (time.time() - pipeline_start) * 1000
                        pipeline_span.set_attribute("pipeline.duration_ms", duration_ms)
                        pipeline_span.set_attribute("pipeline.num_sources", len(literature))
                        pipeline_span.set_attribute("pipeline.confidence", confidence)

                        logger.info(
                            "analyze_xray_completed",
                            use_rag=True,
                            num_sources=len(literature),
                            confidence=confidence,
                            duration_ms=duration_ms,
                        )

                        pipeline_run.end(
                            outputs={
                                "use_rag": True,
                                "num_sources": len(literature),
                                "confidence": confidence,
                                "duration_ms": duration_ms,
                            }
                        )

                        return result

                    except Exception as e:
                        duration_ms = (time.time() - pipeline_start) * 1000
                        logger.error(
                            "analyze_xray_failed",
                            error=str(e),
                            duration_ms=duration_ms,
                        )
                        pipeline_run.end(error=str(e))
                        raise

    def _format_literature(self, literature: List[Dict]) -> str:
        formatted = []
        for i, lit in enumerate(literature):
            source = lit["metadata"].get("source", "Unknown")
            score = lit.get("relevance_score", 0)
            text = lit["text"][:500]
            formatted.append(f"[Source {i+1}] {source} (Relevance: {score:.2f})\n{text}...\n")
        return "\n".join(formatted)

    def _calculate_confidence(self, literature: List[Dict]) -> float:
        if not literature:
            return 0.5
        avg_score = sum(lit.get("relevance_score", 0) for lit in literature) / len(literature)
        return min(0.5 + (avg_score * 0.4), 0.95)

    def batch_analyze(
        self,
        image_directory: str,
        output_directory: str = "./data/analysis_results",
        use_rag: bool = True,
    ) -> List[Dict[str, Any]]:
        """Analyze every image in `image_directory`, writing JSON results."""
        os.makedirs(output_directory, exist_ok=True)

        image_files: List[Path] = []
        for ext in ["*.dcm", "*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(Path(image_directory).glob(ext))

        logger.info(f"Batch analyzing {len(image_files)} images...")

        results = []
        for image_file in image_files:
            try:
                logger.info(f"Processing: {image_file.name}")
                result = self.analyze_xray(str(image_file), use_rag=use_rag)

                output_file = Path(output_directory) / f"{image_file.stem}_analysis.json"
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)

                results.append(
                    {
                        "file": image_file.name,
                        "status": "success",
                        "confidence": result.get("confidence", 0),
                        "output_file": str(output_file),
                    }
                )
                logger.info(f"Analysis saved to {output_file}")

            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
                results.append({"file": image_file.name, "status": "error", "error": str(e)})

        summary_file = Path(output_directory) / "batch_summary.json"
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Batch analysis complete. Summary: {summary_file}")
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store": self.vector_store.get_stats(),
            "model": self.model_name,
        }

"""DocLing-based PDF processing for medical literature."""

import json
from pathlib import Path
from typing import Any, Dict, List

from docling.document_converter import DocumentConverter

from medexrag.observability import get_logger

logger = get_logger(__name__)


class DocLingProcessor:
    """Process medical PDFs using DocLing."""

    def __init__(self, enable_ocr: bool = True):
        logger.info("Initializing DocLing processor...")
        self.converter = DocumentConverter()
        self.enable_ocr = enable_ocr
        logger.info("DocLing processor ready")

    def process_pdf(
        self,
        pdf_path: str,
        save_intermediate: bool = False,
        output_dir: str = None,
    ) -> Dict[str, Any]:
        """Convert one PDF to text + tables + metadata."""
        logger.info(f"Processing PDF: {pdf_path}")

        try:
            result = self.converter.convert(pdf_path)
            text = result.document.export_to_markdown()

            tables = []
            for item, _level in result.document.iterate_items():
                if hasattr(item, "self_ref") and item.self_ref.startswith("#/tables/"):
                    tables.append(
                        {
                            "data": (
                                item.export_to_dataframe().to_dict() if hasattr(item, "export_to_dataframe") else {}
                            ),
                            "caption": getattr(item, "caption", ""),
                        }
                    )

            document = {
                "source": Path(pdf_path).name,
                "text": text,
                "tables": tables,
                "metadata": {
                    "title": result.document.name,
                    "num_pages": len(result.document.pages),
                    "path": pdf_path,
                },
            }

            logger.info(f"Extracted {len(text)} characters from {pdf_path}")

            if save_intermediate:
                out = Path(output_dir) if output_dir else Path(pdf_path).parent
                stem = Path(pdf_path).stem
                md_path = out / f"{stem}_parsed.md"
                json_path = out / f"{stem}_parsed.json"
                md_path.write_text(text, encoding="utf-8")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(document, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved intermediate output to {md_path} and {json_path}")

            return document

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            raise

    def process_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Process every PDF in `directory`."""
        pdf_files = list(Path(directory).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        documents = []
        for pdf_file in pdf_files:
            try:
                documents.append(self.process_pdf(str(pdf_file)))
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")

        logger.info(f"Successfully processed {len(documents)}/{len(pdf_files)} PDFs")
        return documents

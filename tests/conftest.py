"""
Pytest Configuration and Fixtures

This file contains shared fixtures used across all tests.
Fixtures are reusable components that set up test prerequisites.

Learn more: https://docs.pytest.org/en/stable/fixture.html
"""

import pytest
from unittest.mock import MagicMock


# =============================================================================
# Mock Fixtures - Used to avoid loading heavy ML models in CI
# =============================================================================

@pytest.fixture
def mock_vlm():
    """
    Mock VLM (Vision Language Model) for testing without loading real models.

    Real VLM requires:
    - ~4GB model download
    - GPU or significant CPU resources
    - Several minutes to load

    This mock allows testing tool logic without those requirements.
    """
    mock = MagicMock()
    mock.generate.return_value = "Mock analysis: Normal chest X-ray with no acute findings."
    return mock


@pytest.fixture
def mock_embeddings():
    """
    Mock embeddings model for testing without loading PubMedBERT.

    Real embeddings require:
    - ~400MB model download
    - Significant memory

    This mock returns fake embeddings for testing vector store logic.
    """
    mock = MagicMock()
    # Return a fake 768-dimensional embedding (PubMedBERT dimension)
    mock.embed_documents.return_value = [[0.1] * 768]
    mock.embed_query.return_value = [0.1] * 768
    return mock


@pytest.fixture
def sample_document():
    """
    Sample medical document for testing ingestion pipeline.
    """
    return {
        "source": "test_medical_paper.pdf",
        "text": """
        Chest X-Ray Interpretation Guidelines

        Pneumonia presents as consolidation on chest radiographs.
        Key findings include air bronchograms and lobar distribution.

        Differential diagnosis includes:
        - Bacterial pneumonia
        - Viral pneumonia
        - Atelectasis
        """,
        "metadata": {
            "title": "Test Medical Paper",
            "num_pages": 5,
            "path": "/test/path/test_medical_paper.pdf"
        },
        "tables": []
    }


@pytest.fixture
def sample_xray_analysis():
    """
    Sample X-ray analysis result for testing agent workflows.
    """
    return {
        "initial_findings": "Right lower lobe consolidation with air bronchograms",
        "literature_context": "[1] Pneumonia guidelines recommend...",
        "diagnostic_reasoning": "Findings consistent with bacterial pneumonia",
        "confidence": 0.85
    }


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_kb_dir(tmp_path):
    """
    Temporary directory for knowledge base testing.
    Automatically cleaned up after test.
    """
    kb_dir = tmp_path / "test_medical_kb"
    kb_dir.mkdir()
    return str(kb_dir)


@pytest.fixture
def temp_literature_dir(tmp_path):
    """
    Temporary directory for literature files.
    """
    lit_dir = tmp_path / "test_literature"
    lit_dir.mkdir()
    return str(lit_dir)

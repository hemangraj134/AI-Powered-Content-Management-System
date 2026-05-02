"""
test_processing.py – Unit tests for MetaMinds/processing.py.

The SentenceTransformer model and ChromaDB are mocked via conftest.py so these
tests run without a GPU and without downloading any AI artefacts.
"""

import pytest

# conftest.py has already added MetaMinds/ to sys.path and mocked
# sentence_transformers, so this import is safe and fast.
from processing import extract_text_from_txt, process_document


def test_extract_text_from_txt(tmp_path):
    """Plain-text extraction should return the exact file contents."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("This is a test document about AI-powered content management.")

    result = extract_text_from_txt(str(sample_file))

    assert result is not None, "extract_text_from_txt returned None for a valid .txt file."
    assert "AI-powered" in result, "Extracted text does not match the source file content."


def test_extract_text_from_txt_empty_file(tmp_path):
    """An empty .txt file should return an empty string, not None."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    result = extract_text_from_txt(str(empty_file))

    assert result == "", "Expected an empty string for an empty .txt file."


def test_process_document_txt_pipeline(tmp_path):
    """
    End-to-end pipeline test: process_document should return extracted text
    and a valid embedding vector for a standard .txt file.
    The AI model is mocked, so the embedding is a deterministic 384-dim list.
    """
    pipeline_file = tmp_path / "pipeline_test.txt"
    pipeline_file.write_text("Processing pipeline test content for MetaMinds.")

    text, embedding = process_document(str(pipeline_file))

    assert text is not None, "process_document returned None for text on a valid file."
    assert embedding is not None, "process_document returned None for embedding."
    assert len(embedding) == 384, "Embedding vector should have 384 dimensions."


def test_process_document_unsupported_type(tmp_path):
    """process_document should return (None, None) for unsupported file types."""
    bad_file = tmp_path / "archive.zip"
    bad_file.write_text("not a real zip")

    text, embedding = process_document(str(bad_file))

    assert text is None, "Expected None text for an unsupported file type."
    assert embedding is None, "Expected None embedding for an unsupported file type."

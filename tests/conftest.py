"""
conftest.py – PyTest configuration and shared fixtures.

This file is loaded by PyTest BEFORE any test module is imported, which makes
it the correct place to:
  1. Add MetaMinds/ to sys.path so that main.py / processing.py / database.py
     can be imported with their native (non-package) import style.
  2. Mock heavy third-party dependencies (SentenceTransformer, ChromaDB) so
     that tests run fast without GPU, without downloading the ~90 MB AI model,
     and without writing persistent files to disk.
  3. Mock binary/native dependencies (PyMuPDF, python-docx, pytesseract) that
     are imported at module level in processing.py so the test suite is
     self-contained and does not require the full production stack to be
     installed locally.
"""

import os
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Ensure MetaMinds/ modules are importable from the test runner's CWD
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MetaMinds"))

# ---------------------------------------------------------------------------
# 2. Mock sentence_transformers BEFORE processing.py is imported.
#    processing.py calls SentenceTransformer(...) at module level, which would
#    otherwise try to download a ~90 MB model from HuggingFace.
# ---------------------------------------------------------------------------
_mock_encode_result = MagicMock()
_mock_encode_result.tolist.return_value = [0.1] * 384  # Fake 384-dim embedding

_mock_model = MagicMock()
_mock_model.encode.return_value = _mock_encode_result

_mock_st = MagicMock()
_mock_st.SentenceTransformer.return_value = _mock_model

sys.modules["sentence_transformers"] = _mock_st

# ---------------------------------------------------------------------------
# 3. Mock chromadb BEFORE database.py is imported.
#    database.py creates a PersistentClient (writes to disk) at module level.
# ---------------------------------------------------------------------------
_mock_collection = MagicMock()
_mock_collection.query.return_value = {
    "ids": [[]],
    "metadatas": [[]],
    "distances": [[]],
}

_mock_chroma_client = MagicMock()
_mock_chroma_client.get_or_create_collection.return_value = _mock_collection

_mock_chromadb = MagicMock()
_mock_chromadb.PersistentClient.return_value = _mock_chroma_client

sys.modules["chromadb"] = _mock_chromadb

# ---------------------------------------------------------------------------
# 4. Mock native/binary dependencies imported at the top of processing.py.
#    - fitz      (PyMuPDF)   — requires a compiled C extension
#    - docx      (python-docx) — optional in minimal installs
#    - pytesseract             — requires the Tesseract binary
#    Our tests only exercise the plain-text (.txt) path, so these mocks are
#    never actually called; they just need to be importable.
# ---------------------------------------------------------------------------
sys.modules.setdefault("fitz", MagicMock())
sys.modules.setdefault("docx", MagicMock())
sys.modules.setdefault("pytesseract", MagicMock())

# PIL (Pillow) — image processing library, needed by pytesseract in processing.py
_mock_pil = MagicMock()
sys.modules.setdefault("PIL", _mock_pil)
sys.modules.setdefault("PIL.Image", _mock_pil)

# uvicorn — only needed to *run* the server, not to test it. Mock it so
# test_api.py can import main.py without uvicorn being installed.
sys.modules.setdefault("uvicorn", MagicMock())

# torch — processing.py calls torch.cuda.is_available() at module level.
# We mock it to return False (no GPU) so tests are deterministic everywhere.
_mock_torch = MagicMock()
_mock_torch.cuda.is_available.return_value = False
sys.modules.setdefault("torch", _mock_torch)


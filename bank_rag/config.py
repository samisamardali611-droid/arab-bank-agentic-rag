"""Shared paths and model configuration, independent of the working directory."""

from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env", override=False)

DATA_DIR = ROOT_DIR / "data"
OCR_DIR = DATA_DIR / "ocr"
CLEAN_DIR = DATA_DIR / "clean"
SECTIONS_DIR = DATA_DIR / "sections"
CHUNKS_DIR = DATA_DIR / "chunks"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
INDEX_FILE = ARTIFACTS_DIR / "faiss.index"
META_FILE = ARTIFACTS_DIR / "faiss_metadata.json"

# Preserve the submitted models. Changing the embedding model requires rebuilding
# the index with the same model and dimensions used for query embeddings.
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1-mini"

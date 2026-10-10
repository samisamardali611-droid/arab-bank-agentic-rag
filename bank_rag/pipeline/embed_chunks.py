import json
import numpy as np
import faiss
from pathlib import Path
from bank_rag.client import get_client
from tqdm import tqdm

# =========================
# Configuration
# =========================
from bank_rag.config import CHUNKS_DIR, ARTIFACTS_DIR, INDEX_FILE, META_FILE, EMBEDDING_MODEL

CHUNKS_FILE = CHUNKS_DIR / "all_chunks.json"

# =========================
# Main
# =========================
def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("Run the cleaning, sectioning, chunking, and merge stages first.")

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    if not chunks:
        raise ValueError("No chunks to embed.")
    client = get_client()
    print(f"📄 Loaded {len(chunks)} chunks")
    print("🔄 Generating embeddings...")

    texts = [c["text"] for c in chunks]

    embeddings = []
    metadata = []

    for i in tqdm(range(len(texts))):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[i]
        )

        vector = np.array(response.data[0].embedding, dtype="float32")
        embeddings.append(vector)

        # ⭐️ METADATA WITH TEXT ⭐️
        metadata.append({
            "year": chunks[i].get("year"),
            "section_title": chunks[i].get("section_title"),
            "note_number": chunks[i].get("note_number"),
            "text": chunks[i]["text"]   # ← هذا أهم سطر
        })

    embeddings = np.vstack(embeddings)

    # =========================
    # Build FAISS index
    # =========================
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ Embeddings finished successfully")
    print(f"📦 FAISS index saved to: {INDEX_FILE}")
    print(f"📄 Metadata saved to: {META_FILE}")
    print(f"🔢 Total vectors: {index.ntotal}")

# =========================
if __name__ == "__main__":
    main()

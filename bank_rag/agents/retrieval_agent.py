# retrieval_agent.py

import json
import faiss
import numpy as np
from pathlib import Path
from agents import Agent, function_tool
from bank_rag.client import get_client

from bank_rag.engine.reranker import rerank_chunks

# =========================
# Configuration
# =========================
from bank_rag.config import INDEX_FILE, META_FILE, EMBEDDING_MODEL, LLM_MODEL



TOP_K_WIDE = 80     # استرجاع واسع
TOP_K_FINAL = 12    # بعد الريرانك (مهم لتقليل التوكنز)

MAX_CHARS_PER_CHUNK = 1200  # Bound excerpt length; API/token limits can still occur.



# =========================
# Helpers
# =========================
def embed_text(text: str) -> np.ndarray:
    emb = get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(emb.data[0].embedding, dtype="float32").reshape(1, -1)


def load_index():
    index = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
    return index, metadata


# =========================
# Retrieval Tool
# =========================
@function_tool(
    name_override="retrieve_relevant_chunks",
    description_override=(
        "Retrieve and rerank relevant chunks from Arab Bank financial statements "
        "using hybrid retrieval (vector + keyword reranking)."
    ),
)
def retrieve_relevant_chunks(question: str) -> str:
    index, metadata = load_index()

    # ---- Vector Search (Wide) ----
    q_emb = embed_text(question)
    distances, indices = index.search(q_emb, TOP_K_WIDE)

    chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        item = dict(metadata[idx])
        # تحويل المسافة إلى score
        item["score"] = 1 / (1 + float(dist))
        chunks.append(item)

    # ---- Reranking (Hybrid) ----
    chunks = rerank_chunks(chunks, question, top_n=TOP_K_FINAL)

    # ---- Truncate Context (VERY IMPORTANT) ----
    # نرجّع فقط أفضل المقاطع وبنص مقصوص
    return "\n\n".join(
        f"[{c.get('year')} | {c.get('section_title')}]\n"
        f"{(c.get('text') or '')[:MAX_CHARS_PER_CHUNK]}"
        for c in chunks
    )


# =========================
# Agent Definition
# =========================
retrieval_agent = Agent(
    model=LLM_MODEL,
    name="RetrievalAgent",
    instructions=(
        "Retrieve the most relevant evidence from financial statements.\n"
        "Use hybrid retrieval and keep the context concise.\n"
        "Do NOT include unnecessary text."
    ),
    tools=[retrieve_relevant_chunks],
)

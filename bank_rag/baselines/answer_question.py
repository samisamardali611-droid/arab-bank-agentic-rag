import json
import os
import re
import faiss
import numpy as np
from pathlib import Path
from bank_rag.client import get_client

# =========================
# Configuration
# =========================
from bank_rag.config import INDEX_FILE, META_FILE, EMBEDDING_MODEL, LLM_MODEL




TOP_K = 8



# =========================
# Utilities
# =========================
def load_index():
    index = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
    return index, metadata


def embed_text(text: str) -> np.ndarray:
    emb = get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    ).data[0].embedding
    return np.array(emb, dtype="float32").reshape(1, -1)


def detect_year(question: str):
    m = re.search(r"(20\d{2})", question)
    return int(m.group(1)) if m else None


def detect_numeric_question(question: str) -> bool:
    keywords = [
        "total", "amount", "how much", "value",
        "assets", "liabilities", "equity",
        "profit", "loss", "income", "revenue"
    ]
    q = question.lower()
    return any(k in q for k in keywords)


# =========================
# Retrieval
# =========================
def retrieve_chunks(question: str, year_filter=None):
    index, metadata = load_index()
    q_emb = embed_text(question)
    distances, indices = index.search(q_emb, TOP_K)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx]
        item = dict(item)
        item["distance"] = float(dist)
        item["score"] = 1 / (1 + dist)

        if year_filter and item.get("year") != year_filter:
            continue

        results.append(item)

    return results


# =========================
# LLM Answering
# =========================
def llm_answer(question: str, context_items: list) -> str:
    context_text = "\n\n".join(
        f"[Source {i+1} | Year {c.get('year')} | {c.get('section_title')}]\n{c.get('text')}"
        for i, c in enumerate(context_items)
    )

    system_prompt = (
        "You are a financial analyst assistant.\n"
        "Answer the question using ONLY the provided sources.\n"
        "If a numeric value is requested, return the exact number and unit.\n"
        "Do not guess. Do not invent data.\n"
        "Write in professional, concise English."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Sources:\n{context_text}\n\n"
        "Answer:"
    )

    resp = get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return resp.choices[0].message.content.strip()


# =========================
# Formatting
# =========================
def format_professional_answer(question, answer, sources, confidence):
    lines = []
    lines.append("=" * 60)
    lines.append("Arab Bank Financial Statements – Answer")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Question:")
    lines.append(question)
    lines.append("")
    lines.append("Answer:")
    lines.append(answer)
    lines.append("")
    lines.append("Sources:")
    for s in sources:
        lines.append(
            f"- Year: {s.get('year')} | Section: {s.get('section_title')} | Note: {s.get('note_number')}"
        )
    lines.append("")
    lines.append("Confidence:")
    lines.append(confidence)
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# =========================
# Main Logic
# =========================
def answer_question(question: str) -> str:
    year = detect_year(question)
    numeric = detect_numeric_question(question)

    retrieved = retrieve_chunks(question, year_filter=year)

    if not retrieved:
        return (
            "NO ANSWER FOUND\n\n"
            "Reason: No relevant sections matched the question.\n"
            "Suggestion: Try broader wording or remove the year constraint."
        )

    # Use top chunks as context
    context_items = retrieved[:4]

    answer = llm_answer(question, context_items)

    confidence = (
        "High (direct numeric match found in financial statements)"
        if numeric else
        "Medium (answer based on descriptive financial disclosures)"
    )

    return format_professional_answer(
        question=question,
        answer=answer,
        sources=context_items,
        confidence=confidence
    )


# =========================
# CLI
# =========================
def main():
    print("=" * 70)
    print("Arab Bank Financial Statements – Universal Q&A (RAG + LLM)")
    print("=" * 70)

    q = input("Enter your question (English recommended): ").strip()
    if not q:
        print("Empty question. Exiting.")
        return

    print("\n" + answer_question(q))


if __name__ == "__main__":
    main()

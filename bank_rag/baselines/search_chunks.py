import json
import re
import faiss
import numpy as np
from pathlib import Path
from bank_rag.client import get_client

# =========================
# CONFIG
# =========================
from bank_rag.config import INDEX_FILE, META_FILE, EMBEDDING_MODEL, LLM_MODEL

TOP_K = 10

ALLOWED_FINANCIAL_SECTIONS = [
    "statement of financial position",
    "consolidated statement of financial position",
    "financial position",
    "balance sheet"
]



# =========================
# HELPERS
# =========================
def embed(text: str) -> np.ndarray:
    resp = get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(resp.data[0].embedding, dtype="float32")

def extract_year(question: str):
    m = re.search(r"(20\d{2})", question)
    return int(m.group(1)) if m else None

def is_valid_financial_section(section: str) -> bool:
    if not section:
        return False
    s = section.lower()
    return any(key in s for key in ALLOWED_FINANCIAL_SECTIONS)

def looks_like_total_assets_question(q: str) -> bool:
    q = q.lower()
    return "total assets" in q

def extract_number_from_text(text: str):
    """
    استخراج رقم كبير (USD) من النص
    """
    matches = re.findall(r"\b\d{1,3}(?:[\s,]\d{3})+(?:\.\d+)?\b", text)
    if not matches:
        return None
    # نأخذ أكبر رقم
    def normalize(x):
        return float(x.replace(",", "").replace(" ", ""))
    return max(matches, key=lambda x: normalize(x))

# =========================
# CORE
# =========================
def answer_question(question: str) -> str:
    year_filter = extract_year(question)
    is_assets_q = looks_like_total_assets_question(question)

    q_emb = embed(question).reshape(1, -1)

    index = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(META_FILE.read_text(encoding="utf-8"))

    distances, indices = index.search(q_emb, TOP_K)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        meta["distance"] = float(dist)
        candidates.append(meta)

    # =========================
    # FILTERING
    # =========================
    filtered = []

    for c in candidates:
        # سنة
        if year_filter and c.get("year") != year_filter:
            continue

        # سؤال رقمي → لازم قسم مالي
        if is_assets_q:
            if not is_valid_financial_section(c.get("section_title", "")):
                continue

        filtered.append(c)

    if not filtered:
        return (
            "NOT FOUND\n"
            "Reason: No valid financial statement section matched the question.\n"
            f"Year filter: {year_filter}"
        )

    # =========================
    # NUMERIC EXTRACTION
    # =========================
    if is_assets_q:
        for c in filtered:
            number = extract_number_from_text(c.get("text", ""))
            if number:
                return (
                    f"The total assets of Arab Bank Group in {year_filter} were USD {number}.\n\n"
                    f"Source:\n"
                    f"- Year: {c.get('year')}\n"
                    f"- Section: {c.get('section_title')}\n"
                    f"- Note: {c.get('note_number')}"
                )

        return (
            "NOT FOUND\n"
            "Reason: Financial section found, but no Total Assets value could be extracted."
        )

    # =========================
    # FALLBACK (non-numeric)
    # =========================
    best = filtered[0]
    preview = best.get("text", "")[:800]

    return (
        "Answer (contextual):\n\n"
        f"{preview}\n\n"
        "Source:\n"
        f"- Year: {best.get('year')}\n"
        f"- Section: {best.get('section_title')}"
    )

# =========================
# CLI
# =========================
def main():
    print("=" * 70)
    print("Arab Bank Financial Statements – Smart RAG Q&A")
    print("=" * 70)

    q = input("Enter your question (English recommended): ").strip()
    if not q:
        print("Empty question.")
        return

    print("\n" + answer_question(q))

if __name__ == "__main__":
    main()

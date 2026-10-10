import json
import re
import faiss
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from functools import lru_cache
from bank_rag.client import get_client

# =========================
# Configuration
# =========================
from bank_rag.config import INDEX_FILE, META_FILE, EMBEDDING_MODEL, LLM_MODEL






# =========================
# Query Plan + Router
# =========================
@dataclass
class QueryPlan:
    query_type: str                 # numeric | descriptive | comparison
    metric: Optional[str] = None
    years: Optional[List[int]] = None
    needs_verification: bool = False


class QuestionRouter:
    METRICS = {
        "total assets": "total_assets",
        "assets": "total_assets",
        "total liabilities": "total_liabilities",
        "liabilities": "total_liabilities",
        "net profit": "net_profit",
        "profit": "net_profit",
        "loss": "net_profit",
        "total income": "total_income",
        "income": "total_income",
        "revenue": "total_income"
    }

    def route(self, question: str) -> QueryPlan:
        q = question.lower()
        years = sorted({int(y) for y in re.findall(r"(20\d{2})", q)}) or None

        for k, m in self.METRICS.items():
            if k in q:
                if years and len(years) > 1:
                    return QueryPlan("comparison", m, years, True)
                return QueryPlan("numeric", m, years, True)

        return QueryPlan("descriptive", years=years)

# =========================
# Utilities
# =========================
def normalize_number(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    negative = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^\d]", "", s)
    if not s:
        return None
    return f"-{s}" if negative else s


def is_sane_number(value: str, min_digits=6, max_digits=12) -> bool:
    return value.isdigit() and min_digits <= len(value) <= max_digits


@lru_cache(maxsize=1)
def load_index():
    index = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
    return index, metadata


def embed_text(text: str) -> np.ndarray:
    emb = get_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(emb.data[0].embedding, dtype="float32").reshape(1, -1)

# =========================
# Retrieval
# =========================
def adaptive_top_k(plan: QueryPlan) -> int:
    if plan.query_type == "numeric":
        return 4
    if plan.query_type == "comparison":
        return 5
    return 10


def retrieve_chunks(question: str, year=None, top_k=8):
    index, metadata = load_index()
    q_emb = embed_text(question)
    distances, indices = index.search(q_emb, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = dict(metadata[idx])
        if year and item.get("year") != year:
            continue
        item["score"] = 1 / (1 + dist)
        results.append(item)
    return results


def filter_context(chunks: List[Dict], question: str):
    q_words = set(re.findall(r"\w+", question.lower()))
    filtered = [
        c for c in chunks
        if c.get("score", 0) > 0.15 and any(w in c.get("text", "").lower() for w in list(q_words)[:6])
    ]
    return filtered or chunks[:3]

# =========================
# Numeric Verification
# =========================
PATTERNS = {
    "total_assets": r"(total\s+assets)\s*[:\-]?\s*([\d,\s]+)",
    "total_liabilities": r"(total\s+liabilities)\s*[:\-]?\s*([\d,\s]+)",
    "net_profit": r"(net\s+profit).*?([\d,\s\(\)]+)",
    "total_income": r"(total\s+income).*?([\d,\s]+)"
}


def extract_verified_number(metric: str, chunks: List[Dict]) -> Tuple[Optional[str], Optional[List[Dict]]]:
    matches = []
    pattern = PATTERNS.get(metric)
    if not pattern:
        return None, None

    for c in chunks:
        m = re.search(pattern, c.get("text", ""), re.IGNORECASE)
        if m:
            value = normalize_number(m.group(2))
            if value and is_sane_number(value):
                matches.append((value, c))

    if not matches:
        return None, None

    freq = {}
    for v, _ in matches:
        freq[v] = freq.get(v, 0) + 1

    best = max(freq, key=freq.get)
    sources = [c for v, c in matches if v == best][:3]
    return best, sources

# =========================
# LLM
# =========================
def llm_answer(question: str, context: List[Dict]) -> str:
    ctx = "\n\n".join(f"[{c.get('year')} | {c.get('section_title')}]\n{c.get('text')}" for c in context)
    resp = get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Answer strictly from sources. Do not guess."},
            {"role": "user", "content": f"{question}\n\nSources:\n{ctx}"}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# =========================
# Formatting
# =========================
def format_answer(q, a, src, conf):
    out = [
        "=" * 60,
        "Arab Bank Financial Statements – Answer",
        "=" * 60,
        "\nQuestion:\n" + q,
        "\nAnswer:\n" + a,
        "\nSources:"
    ]
    for s in src or []:
        out.append(f"- {s.get('year')} | {s.get('section_title')} | Note {s.get('note_number')}")
    out.append("\nConfidence:\n" + conf)
    out.append("=" * 60)
    return "\n".join(out)

# =========================
# Executors
# =========================
class NumericExecutor:
    def run(self, q, plan):
        k = adaptive_top_k(plan)
        chunks = retrieve_chunks(q, plan.years[0] if plan.years else None, k)
        chunks = filter_context(chunks, q)

        val, src = extract_verified_number(plan.metric, chunks)
        if val:
            return format_answer(q, f"The {plan.metric.replace('_',' ')} is {val}.", src, "High (sanity-verified)")
        return format_answer(q, "No unambiguous numeric value found.", chunks[:3], "Low (OCR ambiguity)")


class DescriptiveExecutor:
    def run(self, q, plan):
        k = adaptive_top_k(plan)
        chunks = retrieve_chunks(q, plan.years[0] if plan.years else None, k)
        chunks = filter_context(chunks, q)
        ans = llm_answer(q, chunks[:6])
        return format_answer(q, ans, chunks[:3], "Medium")


class ComparisonExecutor:
    def run(self, q, plan):
        results, sources = [], []
        for y in plan.years:
            chunks = retrieve_chunks(q, y, adaptive_top_k(plan))
            chunks = filter_context(chunks, q)
            val, src = extract_verified_number(plan.metric, chunks)
            if val:
                results.append(f"{y}: {val}")
                sources.extend(src or [])
        if results:
            return format_answer(q, "; ".join(results), sources[:5], "High")
        return format_answer(q, "Comparison not reliable due to OCR ambiguity.", [], "Low")

# =========================
# Entry Point
# =========================
def answer_question(question: str) -> str:
    plan = QuestionRouter().route(question)
    if plan.query_type == "numeric":
        return NumericExecutor().run(question, plan)
    if plan.query_type == "comparison":
        return ComparisonExecutor().run(question, plan)
    return DescriptiveExecutor().run(question, plan)

# reranker.py

import re
from typing import List, Dict

KEYWORDS_BOOST = {
    "net profit": ["profit", "profit for the year", "attributable"],
    "total assets": ["total assets", "assets"],
    "total income": ["total income", "income", "revenue"],
    "earnings per share": ["earnings per share", "eps", "per share"],
}

def keyword_score(text: str, query: str) -> float:
    t = text.lower()
    q = query.lower()
    score = 0.0

    # exact query words
    for w in q.split():
        if w in t:
            score += 0.3

    # boosted financial phrases
    for k, variants in KEYWORDS_BOOST.items():
        if k in q:
            for v in variants:
                if v in t:
                    score += 1.0

    return score


def rerank_chunks(chunks: List[Dict], question: str, top_n=10) -> List[Dict]:
    ranked = []

    for c in chunks:
        vec_score = c.get("score", 0)  # من FAISS
        key_score = keyword_score(c.get("text", ""), question)

        final_score = (0.7 * vec_score) + (0.3 * key_score)
        c["_final_score"] = final_score
        ranked.append(c)

    ranked.sort(key=lambda x: x["_final_score"], reverse=True)
    return ranked[:top_n]

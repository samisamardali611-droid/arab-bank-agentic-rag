# tools.py
from typing import Dict, List, Optional, Tuple, Any

from bank_rag.engine import rag_engine


# -------------------------
# Tool 1: Triage / Routing
# -------------------------
def tool_route_question(question: str) -> Dict[str, Any]:
    """
    Returns a structured QueryPlan (dict) so agents can make decisions.
    """
    plan = rag_engine.QuestionRouter().route(question)
    return {
        "query_type": plan.query_type,
        "metric": plan.metric,
        "years": plan.years,
        "needs_verification": plan.needs_verification,
    }


# -------------------------
# Tool 2: Retrieval
# -------------------------
def tool_retrieve_chunks(
    question: str,
    year: Optional[int] = None,
    top_k: int = 8
) -> List[Dict[str, Any]]:
    """
    Retrieves chunks from FAISS with optional year filter.
    """
    return rag_engine.retrieve_chunks(question, year=year, top_k=top_k)


def tool_filter_context(chunks: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
    """
    Filters retrieved chunks to a smaller, more relevant context.
    """
    return rag_engine.filter_context(chunks, question)


# -------------------------
# Tool 3: Numeric Verification
# -------------------------
def tool_extract_verified_number(
    metric: str,
    chunks: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    Extracts and sanity-verifies a numeric value from retrieved chunks.
    """
    return rag_engine.extract_verified_number(metric, chunks)


# -------------------------
# Tool 4: LLM Answering
# -------------------------
def tool_llm_answer(question: str, context: List[Dict[str, Any]]) -> str:
    """
    Generates an answer strictly from the provided context.
    """
    return rag_engine.llm_answer(question, context)


# -------------------------
# Tool 5: Formatting
# -------------------------
def tool_format_answer(
    question: str,
    answer: str,
    sources: Optional[List[Dict[str, Any]]],
    confidence: str
) -> str:
    """
    Produces final formatted output with sources + confidence.
    """
    return rag_engine.format_answer(question, answer, sources, confidence)


# -------------------------
# Tool 6: Full Pipeline (baseline)
# -------------------------
def tool_answer_full(question: str) -> str:
    """
    Baseline full answer using your current engine.
    This is useful for comparison while we build the agentic layer.
    """
    return rag_engine.answer_question(question)
# --- LLM Planner Tool (General Routing) ---
import json
from bank_rag.client import get_client
from bank_rag.config import LLM_MODEL



def tool_plan_question_llm(question: str):
    """
    General-purpose planner that supports ANY question types.
    Returns:
      {
        "query_type": "numeric" | "descriptive" | "comparison",
        "metric_text": <string or null>,
        "years": [YYYY, ...] or null,
        "needs_verification": true/false
      }
    Fallback: if parsing fails, uses the old deterministic router.
    """
    system = (
        "You are a planner for a financial statements RAG chatbot.\n"
        "Classify the user question into one of: numeric, descriptive, comparison.\n"
        "- numeric: asks for a single value for one year.\n"
        "- comparison: asks to compare values across multiple years.\n"
        "- descriptive: explanation, policy, narrative, definitions.\n"
        "Extract years if present (list of integers). If none, years=null.\n"
        "If numeric/comparison, set needs_verification=true and set metric_text to the financial measure as a short phrase.\n"
        "If descriptive, set needs_verification=false and metric_text=null.\n"
        "Return ONLY valid JSON."
    )

    try:
        resp = get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        plan = json.loads(resp.choices[0].message.content)

        # minimal normalization
        qt = (plan.get("query_type") or "").strip().lower()
        if qt not in {"numeric", "descriptive", "comparison"}:
            raise ValueError("bad query_type")

        years = plan.get("years")
        if years is not None:
            years = [int(y) for y in years if str(y).isdigit()]
            years = sorted(list(set(years))) or None

        metric_text = plan.get("metric_text")
        if qt in {"numeric", "comparison"}:
            metric_text = (metric_text or "").strip() or None
            needs_verification = True
        else:
            metric_text = None
            needs_verification = False

        return {
            "query_type": qt,
            "metric_text": metric_text,
            "years": years,
            "needs_verification": needs_verification,
        }

    except Exception:
        # fallback to your current router
        p = tool_route_question(question)
        return {
            "query_type": p["query_type"],
            "metric_text": p.get("metric") or None,
            "years": p.get("years"),
            "needs_verification": bool(p.get("needs_verification")),
        }

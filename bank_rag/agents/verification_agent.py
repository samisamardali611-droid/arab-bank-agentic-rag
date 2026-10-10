# verification_agent.py

from typing import Dict, Any, Optional
import re
from agents import Agent, function_tool
from bank_rag.config import LLM_MODEL


# ---------- Pure Python (CALL THIS FROM ORCHESTRATOR) ----------
def verify_numeric_from_context_py(metric_text: str, context_text: str) -> Dict[str, Any]:
    """
    Deterministic numeric verification from retrieved context (NO LLM).
    Returns:
      { "status": "verified"|"not_found", "value": <str|None>, "evidence": <str|None> }
    """
    if not metric_text or not context_text:
        return {"status": "not_found", "value": None, "evidence": None}

    metric = " ".join(metric_text.lower().replace("_", " ").split())
    text = context_text

    # Common money / number pattern (handles commas/spaces and parentheses)
    num_pat = r"(\(?\d{1,3}(?:[, ]\d{3})*(?:\.\d+)?\)?)"

    # Metric-aware patterns (add synonyms that appear in statements)
    patterns = []

    if "net profit" in metric or metric == "profit":
        patterns += [
            rf"(net\s+profit).*?{num_pat}",
            rf"(profit\s+for\s+the\s+year).*?{num_pat}",
            rf"(attributable\s+to\s+shareholders).*?{num_pat}",
        ]
    elif "total assets" in metric or metric == "assets":
        patterns += [
            rf"(total\s+assets).*?{num_pat}",
            rf"(assets\s+total).*?{num_pat}",
        ]
    elif "total income" in metric or "income" in metric or "revenue" in metric:
        patterns += [
            rf"(total\s+income).*?{num_pat}",
            rf"(income).*?{num_pat}",
            rf"(revenue).*?{num_pat}",
        ]
    elif "earnings per share" in metric or metric == "eps" or "per share" in metric:
        patterns += [
            rf"(earnings\s+per\s+share).*?{num_pat}",
            rf"(eps).*?{num_pat}",
            rf"(basic\s+and\s+diluted).*?{num_pat}",
        ]
    else:
        # generic fallback: try to find metric words near a number
        # e.g. "total equity .... 123,456"
        key = re.escape(metric)
        patterns += [rf"({key}).*?{num_pat}"]

    # Search
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue

        raw = m.group(2) if m.lastindex and m.lastindex >= 2 else None
        if not raw:
            continue

        value = raw.strip()
        # normalize (remove commas/spaces; keep negative parentheses)
        negative = value.startswith("(") and value.endswith(")")
        digits = re.sub(r"[^\d\.]", "", value)
        if not digits:
            continue

        # sanity: allow decimals for EPS; otherwise expect big-ish numbers
        if "per share" in metric or "eps" in metric:
            ok = True
        else:
            ok = len(re.sub(r"\D", "", digits)) >= 6  # at least 6 digits (USD '000 etc.)

        if not ok:
            continue

        norm = f"-{digits}" if negative else digits

        # Evidence = a short surrounding snippet
        start = max(m.start() - 120, 0)
        end = min(m.end() + 120, len(text))
        evidence = text[start:end].replace("\n", " ").strip()

        return {"status": "verified", "value": norm, "evidence": evidence}

    return {"status": "not_found", "value": None, "evidence": None}


# ---------- Tool wrapper (ONLY FOR AGENT USAGE) ----------
@function_tool(
    name_override="verify_numeric_from_context",
    description_override="Verify a numeric value deterministically from retrieved context."
)
def verify_numeric_from_context_tool(metric_text: str, context_text: str) -> Dict[str, Any]:
    return verify_numeric_from_context_py(metric_text, context_text)


verification_agent = Agent(
    model=LLM_MODEL,
    name="VerificationAgent",
    instructions=(
        "Verify numeric values deterministically from the provided context.\n"
        "Return JSON with status/value/evidence. Never guess."
    ),
    tools=[verify_numeric_from_context_tool],
)

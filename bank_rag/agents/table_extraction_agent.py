# table_extraction_agent.py

from typing import Dict, Any, List, Optional, Tuple
from agents import Agent, function_tool
from bank_rag.config import LLM_MODEL
import re


def normalize_metric(metric_text: str) -> str:
    if not metric_text:
        return ""
    return " ".join(metric_text.lower().replace("_", " ").split())


def metric_variants(metric_text: str) -> List[str]:
    m = normalize_metric(metric_text)
    variants = {m}

    if "net profit" in m or m == "profit":
        variants.update([
            "net profit",
            "profit for the year",
            "attributable to shareholders",
            "profit attributable",
            "profit",
        ])

    if "total assets" in m or m == "assets":
        variants.update(["total assets", "assets"])

    if "total income" in m or "income" in m or "revenue" in m:
        variants.update(["total income", "income", "revenue"])

    if "equity" in m:
        variants.update(["equity", "total equity", "shareholders' equity"])

    if "earnings per share" in m or "eps" in m or "per share" in m:
        variants.update(["earnings per share", "eps", "per share", "basic and diluted"])

    return list(variants)


def extract_numbers_with_positions(line: str) -> List[Tuple[str, int]]:
    number_pattern = r"\b\d{1,3}(?:[, ]\d{3})*(?:\.\d+)?\b"
    out = []
    for m in re.finditer(number_pattern, line):
        raw = m.group(0)
        clean = raw.replace(",", "").replace(" ", "")
        out.append((clean, m.start()))
    return out


def find_header_years(lines: List[str], target_year: int) -> Optional[List[int]]:
    """
    Find a line that contains multiple years including target_year, treat it as header order.
    """
    for line in lines:
        yrs = [int(y) for y in re.findall(r"\b20\d{2}\b", line)]
        if target_year in yrs and len(yrs) >= 2:
            seen = []
            for y in yrs:
                if y not in seen:
                    seen.append(y)
            return seen
    return None


# ---------- Pure Python (CALL THIS FROM ORCHESTRATOR) ----------
def extract_numeric_from_tables_py(metric_text: str, year: int, context_text: str) -> Dict[str, Any]:
    metric_text = normalize_metric(metric_text)
    variants = metric_variants(metric_text)

    lines = context_text.splitlines()
    header_years = find_header_years(lines, year)

    candidates = []
    for ln in lines:
        lnl = ln.lower()
        if any(v in lnl for v in variants) and re.search(r"\d", ln):
            candidates.append(ln)

    if not candidates:
        return {"status": "not_found", "value": None, "evidence": None}

    # Year-aware selection if header exists
    if header_years and year in header_years:
        y_idx = header_years.index(year)

        for ln in candidates:
            nums = extract_numbers_with_positions(ln)
            # remove any year tokens on the row
            nums = [(n, pos) for (n, pos) in nums if not (len(n) == 4 and n.startswith("20"))]

            if len(nums) >= len(header_years):
                nums_sorted = sorted(nums, key=lambda x: x[1])
                picked = nums_sorted[y_idx][0]
                return {"status": "verified_from_table", "value": picked, "evidence": ln.strip()}

    # Fallback: pick the most amount-like number
    all_found = []
    for ln in candidates:
        nums = [n for (n, _) in extract_numbers_with_positions(ln)]
        nums = [n for n in nums if not (len(n) == 4 and n.startswith("20"))]
        for n in nums:
            all_found.append((n, ln.strip()))

    if not all_found:
        return {"status": "not_found", "value": None, "evidence": None}

    # Prefer decimals for EPS
    if ("per share" in metric_text) or ("eps" in metric_text) or ("earnings per share" in metric_text):
        all_found.sort(key=lambda x: ("." not in x[0], -len(x[0])))
    else:
        all_found.sort(key=lambda x: (-len(x[0]), x[0].count(".")))

    best_val, best_line = all_found[0]
    return {"status": "verified_from_table", "value": best_val, "evidence": best_line}


# ---------- Tool wrapper (ONLY FOR AGENT USAGE) ----------
@function_tool(
    name_override="extract_numeric_from_tables",
    description_override="Extract a numeric value from table-like context using year-aware heuristics."
)
def extract_numeric_from_tables_tool(metric_text: str, year: int, context_text: str) -> Dict[str, Any]:
    return extract_numeric_from_tables_py(metric_text, year, context_text)


table_extraction_agent = Agent(
    model=LLM_MODEL,
    name="TableExtractionAgent",
    instructions="Extract numeric values from table-like text conservatively. Never guess.",
    tools=[extract_numeric_from_tables_tool],
)

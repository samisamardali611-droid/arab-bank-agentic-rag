# orchestrator.py

import json
import ast
from agents import Runner

from bank_rag.agents.triage_agent import triage_agent
from bank_rag.agents.retrieval_agent import retrieval_agent

from bank_rag.engine import rag_engine

from bank_rag.agents.verification_agent import verify_numeric_from_context_py
from bank_rag.agents.table_extraction_agent import extract_numeric_from_tables_py


def safe_parse(output):
    if isinstance(output, dict):
        return output
    if not isinstance(output, str):
        return {}
    try:
        parsed = json.loads(output)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(output)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def normalize_metric_text(m: str) -> str:
    if not m:
        return ""
    return " ".join(m.lower().replace("_", " ").split())


async def run_orchestrator(question: str) -> str:
    runner = Runner()

    # 1) Planning
    triage_res = await runner.run(triage_agent, input=question)
    plan = safe_parse(triage_res.final_output)

    query_type = plan.get("query_type")
    metric_text = normalize_metric_text(plan.get("metric_text", "") or plan.get("metric", ""))
    years = plan.get("years") or []

    # 2) Retrieval
    retrieval_res = await runner.run(retrieval_agent, input=question)
    context_text = retrieval_res.final_output or ""

    # 3) Numeric / comparison
    if query_type in {"numeric", "comparison"} and metric_text:
        ver = verify_numeric_from_context_py(metric_text, context_text)

        if ver.get("status") == "verified":
            ans = f"The {metric_text} is {ver.get('value')}."
            if ver.get("evidence"):
                ans += f"\n\nEvidence:\n{ver.get('evidence')}"
            return rag_engine.format_answer(question, ans, [], "High (explicit textual evidence)")

        if years:
            tab = extract_numeric_from_tables_py(metric_text, int(years[0]), context_text)
            if tab.get("status") == "verified_from_table":
                ans = f"The {metric_text} in {years[0]} is {tab.get('value')} (from tables)."
                if tab.get("evidence"):
                    ans += f"\n\nEvidence:\n{tab.get('evidence')}"
                return rag_engine.format_answer(question, ans, [], "Medium (table-based evidence)")

        return rag_engine.format_answer(
            question,
            "The requested value is not explicitly stated in the available documents.",
            [],
            "Low (no explicit evidence)"
        )

    # 4) Descriptive
    # keep context bounded
    answer = rag_engine.llm_answer(question, [{"text": context_text[:2000]}])
    return rag_engine.format_answer(question, answer, [], "Medium (descriptive answer from sources)")

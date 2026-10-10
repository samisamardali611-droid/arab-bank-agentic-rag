# triage_agent.py
from typing import Any, Dict
from agents import Agent, function_tool
from bank_rag.config import LLM_MODEL
from bank_rag.agents.tools import tool_plan_question_llm


@function_tool(
    name_override="plan_question",
    description_override="Plan a question: decide type, extract years, and identify the metric (if any)."
)
def plan_question(question: str) -> Dict[str, Any]:
    return tool_plan_question_llm(question)


triage_agent = Agent(
    model=LLM_MODEL,
    name="QueryTriageAgent",
    instructions=(
        "You are a routing/planning agent for a financial RAG system.\n"
        "You MUST call the plan_question tool and return its JSON output as-is.\n"
        "Do not answer the user's question."
    ),
    tools=[plan_question],
)

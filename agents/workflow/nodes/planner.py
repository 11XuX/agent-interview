"""Planner：研究问题 -> 子问题 + 检索式。"""

from langchain_core.prompts import ChatPromptTemplate

from ..llm import llm
from ..models import Plan
from ..state import State

chain = ChatPromptTemplate.from_messages([
    ("system", "你是文献调研助手。把研究问题拆成子问题，并为每个子问题写一条检索式。"
               "检索式要收得住 —— 宁可少召回也不要把整个领域都捞进来。"),
    ("human", "{question}"),
]) | llm.with_structured_output(Plan, method="function_calling")


def planner(state: State) -> State:
    """顺便把第一轮的工作队列填上。"""
    plan = chain.invoke({"question": state["question"]})
    return {"plan": plan, "pending": plan.sub_queries, "round": 0}

"""Check：数每个子问题的证据够不够，不够就生成新检索式回到 search。

这个节点和它的条件边是整张图里 LCEL 写不出来的部分 —— 跑几轮
取决于运行时状态，不是编译期决定的。
"""

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END

from ..llm import MAX_CONCURRENCY, llm
from ..models import RetryQuery, SubQuery
from ..state import State

MIN_PAPERS = 2      # 每个子问题至少要有几篇够格的文献
MAX_ROUNDS = 2      # 最多补检索几轮

chain = ChatPromptTemplate.from_messages([
    ("system", "上一条检索式召回的文献不够用。换一个角度重写，比如换同义词、放宽限定、"
               "或者改查这个问题的上位概念。不要和已试过的检索式雷同。"),
    ("human", "子问题：{question}\n\n已试过的检索式：\n{tried}"),
]) | llm.with_structured_output(RetryQuery, method="function_calling")


async def check(state: State) -> State:
    counts = {
        sq.question: sum(1 for p in state["papers"] if sq.question in p.found_for)
        for sq in state["plan"].sub_queries
    }
    gaps = [q for q, n in counts.items() if n < MIN_PAPERS]

    # 到轮次上限就不再补，把 gaps 留给下游说明"这块证据不足"
    if not gaps or state["round"] >= MAX_ROUNDS:
        return {"gaps": gaps}

    tried = "\n".join(f"- {t}" for t in state.get("tried", []))
    retries: list[RetryQuery] = await chain.abatch(
        [{"question": q, "tried": tried} for q in gaps],
        config={"max_concurrency": MAX_CONCURRENCY},
    )
    return {
        "gaps": gaps,
        "round": state["round"] + 1,
        "pending": [SubQuery(question=q, query=r.query) for q, r in zip(gaps, retries)],
    }


def route_after_check(state: State) -> str:
    """条件边：还有 pending 就回 search 再查一轮，否则结束。"""
    return "search" if state.get("pending") else END

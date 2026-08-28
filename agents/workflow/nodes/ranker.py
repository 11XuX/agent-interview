"""Paper Ranker：跨源去重 + 相关性筛选。"""

from langchain_core.prompts import ChatPromptTemplate

from ..llm import MAX_CONCURRENCY, llm
from ..models import Relevance, dedup
from ..state import State

MIN_SCORE = 2       # 低于这个分的丢掉

chain = ChatPromptTemplate.from_messages([
    ("system", "你是文献筛选员。判断这篇文献能不能回答给定的子问题。"
               "只看标题和摘要提供的事实，不要脑补。宁可给低分也不要放过不相关的。"),
    ("human", "子问题：\n{questions}\n\n标题：{title}\n\n摘要：{abstract}"),
]) | llm.with_structured_output(Relevance, method="function_calling")


async def ranker(state: State) -> State:
    """abatch 是 Runnable 基类白送的：传一批输入，内部并发跑，结果按输入顺序返回。"""
    papers = dedup(state["papers"])
    fresh = [p for p in papers if not p.reason]     # 上一轮已打过分的不重复花钱

    if fresh:
        scores: list[Relevance] = await chain.abatch(
            [{"questions": "\n".join(f"- {q}" for q in p.found_for),
              "title": p.title,
              "abstract": p.abstract[:1500] or "（无摘要）"} for p in fresh],
            config={"max_concurrency": MAX_CONCURRENCY},
        )
        for p, r in zip(fresh, scores):     # abatch 保序，可以直接 zip
            p.score, p.reason = r.score, r.reason

    kept = [p for p in papers if p.score >= MIN_SCORE]
    kept.sort(key=lambda p: (-p.score, -len(p.found_for)))
    return {"papers": kept}

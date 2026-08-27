"""节点。每个节点：读状态，返回补丁。"""

import asyncio
import os

import httpx
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .sources import SOURCES
from .state import Plan, Relevance, State, dedup

load_dotenv()

llm = ChatOpenAI(
    model=os.environ["MODEL"],
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
    # DeepSeek V4 默认开思考，而思考模式拒绝强制 tool_choice。
    # 结构化输出走 function_calling 必然下发强制 tool_choice，所以要关掉。
    extra_body={"thinking": {"type": "disabled"}},
)

plan_chain = ChatPromptTemplate.from_messages([
    ("system", "你是文献调研助手。把研究问题拆成子问题，并为每个子问题写一条检索式。"
               "检索式要收得住 —— 宁可少召回也不要把整个领域都捞进来。"),
    ("human", "{question}"),
]) | llm.with_structured_output(Plan, method="function_calling")


def planner(state: State) -> State:
    """研究问题 -> 子问题 + 检索式。"""
    return {"plan": plan_chain.invoke({"question": state["question"]})}


async def search(state: State) -> State:
    """所有 (源 x 检索式) 组合并发查。"""
    async with httpx.AsyncClient(timeout=20) as client:
        batches = await asyncio.gather(*(
            src(client, sq)
            for sq in state["plan"].sub_queries
            for src in SOURCES
        ))
    return {"papers": [p for b in batches for p in b]}


MIN_SCORE = 2        # 低于这个分的丢掉
MAX_CONCURRENCY = 5  # 同时最多几个模型调用

rank_chain = ChatPromptTemplate.from_messages([
    ("system", "你是文献筛选员。判断这篇文献能不能回答给定的子问题。"
               "只看标题和摘要提供的事实，不要脑补。宁可给低分也不要放过不相关的。"),
    ("human", "子问题：\n{questions}\n\n标题：{title}\n\n摘要：{abstract}"),
]) | llm.with_structured_output(Relevance, method="function_calling")


async def ranker(state: State) -> State:
    """去重 + 相关性筛选。

    abatch 是 Runnable 基类白送的：一次传一批输入，内部并发跑，
    结果按输入顺序返回。max_concurrency 限制同时在飞的数量。
    """
    papers = dedup(state["papers"])

    scores: list[Relevance] = await rank_chain.abatch(
        [{"questions": "\n".join(f"- {q}" for q in p.found_for),
          "title": p.title,
          "abstract": p.abstract[:1500] or "（无摘要）"} for p in papers],
        config={"max_concurrency": MAX_CONCURRENCY},
    )

    kept = []
    for p, r in zip(papers, scores):     # abatch 保序，可以直接 zip
        p.score, p.reason = r.score, r.reason
        if r.score >= MIN_SCORE:
            kept.append(p)
    kept.sort(key=lambda p: (-p.score, -len(p.found_for)))
    return {"papers": kept}

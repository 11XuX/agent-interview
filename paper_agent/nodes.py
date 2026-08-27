"""节点。每个节点：读状态，返回补丁。"""

import asyncio
import os

import httpx
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from .sources import SOURCES
from .state import Plan, Relevance, RetryQuery, State, SubQuery, dedup

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
    """研究问题 -> 子问题 + 检索式。顺便把第一轮的工作队列填上。"""
    plan = plan_chain.invoke({"question": state["question"]})
    return {"plan": plan, "pending": plan.sub_queries, "round": 0}


async def search(state: State) -> State:
    """把 pending 里的检索式跑掉，结果**追加**到 papers。

    追加而不是覆盖 —— 循环第二轮进来时，第一轮筛出来的好文献要留着。
    papers 没有 reducer，所以这里显式地读旧值再拼。
    """
    pending = state["pending"]
    async with httpx.AsyncClient(timeout=20) as client:
        batches = await asyncio.gather(*(
            src(client, sq) for sq in pending for src in SOURCES
        ))
    found = [p for b in batches for p in b]
    return {
        "papers": state.get("papers", []) + found,
        "tried": state.get("tried", []) + [sq.query for sq in pending],
        "pending": [],
    }


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
    fresh = [p for p in papers if not p.reason]     # 上一轮已经打过分的不重复花钱

    if fresh:
        scores: list[Relevance] = await rank_chain.abatch(
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


MIN_PAPERS = 2       # 每个子问题至少要有几篇够格的文献
MAX_ROUNDS = 2       # 最多补检索几轮

retry_chain = ChatPromptTemplate.from_messages([
    ("system", "上一条检索式召回的文献不够用。换一个角度重写，比如换同义词、放宽限定、"
               "或者改查这个问题的上位概念。不要和已试过的检索式雷同。"),
    ("human", "子问题：{question}\n\n已试过的检索式：\n{tried}"),
]) | llm.with_structured_output(RetryQuery, method="function_calling")


async def check(state: State) -> State:
    """数每个子问题剩几篇，不够的生成新检索式塞回 pending。"""
    counts = {
        sq.question: sum(1 for p in state["papers"] if sq.question in p.found_for)
        for sq in state["plan"].sub_queries
    }
    gaps = [q for q, n in counts.items() if n < MIN_PAPERS]

    # 到轮次上限就不再补了，把 gaps 记下来交给下游说明"这块证据不足"
    if not gaps or state["round"] >= MAX_ROUNDS:
        return {"gaps": gaps}

    tried = "\n".join(f"- {t}" for t in state.get("tried", []))
    retries: list[RetryQuery] = await retry_chain.abatch(
        [{"question": q, "tried": tried} for q in gaps],
        config={"max_concurrency": MAX_CONCURRENCY},
    )
    return {
        "gaps": gaps,
        "round": state["round"] + 1,
        "pending": [SubQuery(question=q, query=r.query) for q, r in zip(gaps, retries)],
    }


def route_after_check(state: State) -> str:
    """条件边：还有 pending 就回去再查一轮，否则结束。"""
    return "search" if state.get("pending") else END

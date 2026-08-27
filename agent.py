import os
from typing import TypedDict

import httpx
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model=os.environ["MODEL"],
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
)

EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class SubQuery(BaseModel):
    """一个子问题，以及查它该用的检索式。"""

    question: str = Field(description="子问题，中文，必须能被文献证据直接支撑或否定")
    query: str = Field(
        description=(
            "Europe PMC 检索式，英文。核心概念之间用 AND，同义词用 OR 并括起来。"
            '例：(scRNA-seq OR "single-cell RNA-seq") AND "batch effect" AND (benchmark OR comparison)'
        )
    )


class Plan(BaseModel):
    sub_queries: list[SubQuery] = Field(description="3-5 组，覆盖回答原问题所需的各个方面")


class Paper(BaseModel):
    pmcid: str
    title: str
    year: int
    found_for: str          # 哪个子问题查出来的


class State(TypedDict, total=False):
    question: str
    plan: Plan
    papers: list[Paper]


plan_chain = ChatPromptTemplate.from_messages([
    ("system", "你是文献调研助手。把研究问题拆成子问题，并为每个子问题写一条 Europe PMC 检索式。"
               "检索式要收得住 —— 宁可少召回也不要把整个领域都捞进来。"),
    ("human", "{question}"),
]) | llm.with_structured_output(Plan, method="function_calling")


def make_plan(state: State) -> State:
    return {"plan": plan_chain.invoke({"question": state["question"]})}


def search(state: State) -> State:
    """每条检索式查一次。目前是串行，下一步改并发。"""
    papers = []
    with httpx.Client(timeout=20) as client:
        for sq in state["plan"].sub_queries:
            r = client.get(EUROPEPMC, params={
                "query": f"({sq.query}) AND OPEN_ACCESS:Y",
                "format": "json", "pageSize": 3,
            })
            r.raise_for_status()
            for it in r.json()["resultList"]["result"]:
                papers.append(Paper(pmcid=it.get("pmcid", ""), title=it["title"],
                                    year=int(it.get("pubYear") or 0), found_for=sq.question))
    return {"papers": papers}


builder = StateGraph(State)
builder.add_node("make_plan", make_plan)
builder.add_node("search", search)
builder.add_edge(START, "make_plan")
builder.add_edge("make_plan", "search")
builder.add_edge("search", END)
graph = builder.compile()

out = graph.invoke({"question": "单细胞 RNA-seq 的批次效应校正方法哪类更可靠"})
for sq in out["plan"].sub_queries:
    print(f"\n■ {sq.question}")
    print(f"  检索式: {sq.query}")
    for p in out["papers"]:
        if p.found_for == sq.question:
            print(f"    {p.year} {p.pmcid:12} {p.title[:58]}")
print(f"\n共 {len(out['papers'])} 篇")

import os
from typing import TypedDict

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

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

EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class Plan(BaseModel):
    """把研究问题拆成可检索的形式。"""

    keywords: list[str] = Field(description="英文检索关键词，3-5 个")
    sub_questions: list[str] = Field(description="必须回答的子问题，3-5 个，每个都能被文献证据支撑或否定")


class Paper(BaseModel):
    pmcid: str
    title: str
    year: int


class State(TypedDict, total=False):
    question: str
    plan: Plan
    papers: list[Paper]


def make_plan(state: State) -> State:
    p = llm.with_structured_output(Plan, method="function_calling").invoke(state["question"])
    return {"plan": p}


def search(state: State) -> State:
    """拿关键词去查 Europe PMC。节点第一次碰外部世界。"""
    q = " OR ".join(f'"{k}"' for k in state["plan"].keywords)
    r = httpx.get(
        EUROPEPMC,
        params={"query": f"({q}) AND OPEN_ACCESS:Y", "format": "json", "pageSize": 5},
        timeout=20,
    )
    r.raise_for_status()
    papers = [
        Paper(pmcid=it.get("pmcid", ""), title=it["title"], year=int(it.get("pubYear") or 0))
        for it in r.json()["resultList"]["result"]
    ]
    return {"papers": papers}


builder = StateGraph(State)
builder.add_node("make_plan", make_plan)
builder.add_node("search", search)
builder.add_edge(START, "make_plan")
builder.add_edge("make_plan", "search")
builder.add_edge("search", END)
graph = builder.compile()

out = graph.invoke({"question": "单细胞 RNA-seq 的批次效应校正方法哪类更可靠"})
print("关键词:", out["plan"].keywords)
print(f"查到 {len(out['papers'])} 篇:")
for p in out["papers"]:
    print(f"  {p.year}  {p.pmcid:12} {p.title[:60]}")

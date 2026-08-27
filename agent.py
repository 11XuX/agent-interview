import os
from typing import TypedDict

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


class Plan(BaseModel):
    """把研究问题拆成可检索的形式。"""

    keywords: list[str] = Field(description="英文检索关键词，3-5 个")
    sub_questions: list[str] = Field(description="必须回答的子问题，3-5 个，每个都能被文献证据支撑或否定")


class State(TypedDict, total=False):
    question: str
    plan: Plan


def make_plan(state: State) -> State:
    """模型返回 Plan 对象，不是文本。"""
    p = llm.with_structured_output(Plan, method="function_calling").invoke(state["question"])
    return {"plan": p}


builder = StateGraph(State)
builder.add_node("make_plan", make_plan)
builder.add_edge(START, "make_plan")
builder.add_edge("make_plan", END)
graph = builder.compile()

out = graph.invoke({"question": "单细胞 RNA-seq 的批次效应校正方法哪类更可靠"})
p = out["plan"]
print("类型:", type(p).__name__)
print("keywords:", p.keywords)
for q in p.sub_questions:
    print("  -", q)

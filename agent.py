import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

llm = ChatOpenAI(
    model=os.environ["MODEL"],
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
)


class State(TypedDict):
    question: str
    answer: str


def ask(state: State) -> dict:
    """节点里干什么框架不管。这里调一次模型。"""
    resp = llm.invoke(state["question"])       # 返回 AIMessage 对象
    return {"answer": resp.content}            # .content 才是文本


builder = StateGraph(State)
builder.add_node("ask", ask)
builder.add_edge(START, "ask")
builder.add_edge("ask", END)
graph = builder.compile()

out = graph.invoke({"question": "用一句话说清楚什么是批次效应"})
print(out["answer"])

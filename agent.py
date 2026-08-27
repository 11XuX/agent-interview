from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    count: int
    log: Annotated[list[str], list.__add__]


def add_one(state: State) -> dict:
    return {"count": state["count"] + 1, "log": [f"加到 {state['count'] + 1}"]}


def route(state: State) -> str:
    """够了就结束，不够就回 add_one 再来一轮。"""
    return END if state["count"] >= 5 else "add_one"


builder = StateGraph(State)
builder.add_node("add_one", add_one)
builder.add_edge(START, "add_one")
builder.add_conditional_edges("add_one", route)   # ← 指回自己，成环
graph = builder.compile()

out = graph.invoke({"count": 0, "log": []})
print("count:", out["count"])
print("log:  ", out["log"])

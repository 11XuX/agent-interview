from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    count: int


def add_one(state: State) -> dict:
    print(f"  add_one 收到 {state}")
    return {"count": state["count"] + 1}


def double(state: State) -> dict:
    print(f"  double  收到 {state}")
    return {"count": state["count"] * 2}


builder = StateGraph(State)
builder.add_node("add_one", add_one)
builder.add_node("double", double)
builder.add_edge(START, "add_one")
builder.add_edge("add_one", "double")     # ← 新增：add_one 跑完接着跑 double
builder.add_edge("double", END)
graph = builder.compile()

print(graph.invoke({"count": 0}))

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    count: int
    note: str          # ← 新增第二个字段


def add_one(state: State) -> dict:
    print(f"  add_one 收到 {state}")
    return {"count": state["count"] + 1}          # 只改 count，没碰 note


def double(state: State) -> dict:
    print(f"  double  收到 {state}")
    return {"count": state["count"] * 2,
            "note": "double 改过了"}              # 两个字段都改


builder = StateGraph(State)
builder.add_node("add_one", add_one)
builder.add_node("double", double)
builder.add_edge(START, "add_one")
builder.add_edge("add_one", "double")
builder.add_edge("double", END)
graph = builder.compile()

print(graph.invoke({"count": 0, "note": "开工"}))

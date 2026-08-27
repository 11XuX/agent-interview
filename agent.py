from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    count: int
    note: str
    log: Annotated[list[str], list.__add__]   # ← 标了 reducer：写入时拼接，不覆盖


def add_one(state: State) -> dict:
    return {"count": state["count"] + 1,
            "note": "add_one 到此一游",
            "log": ["add_one 跑了"]}          # 注意：要包成列表


def double(state: State) -> dict:
    return {"count": state["count"] * 2,
            "note": "double 到此一游",
            "log": ["double 跑了"]}


builder = StateGraph(State)
builder.add_node("add_one", add_one)
builder.add_node("double", double)
builder.add_edge(START, "add_one")
builder.add_edge("add_one", "double")
builder.add_edge("double", END)
graph = builder.compile()

out = graph.invoke({"count": 0, "note": "开工", "log": ["main 起手"]})
print("count:", out["count"])
print("note: ", out["note"])
print("log:  ", out["log"])

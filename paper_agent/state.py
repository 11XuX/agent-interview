"""图的共享状态。

字段标没标 Annotated[..., reducer] 决定并发写入时是覆盖还是归约。
目前没有字段需要 reducer —— 每个字段都只有一个节点写。
"""

from typing import TypedDict

from .models import Paper, Plan, SubQuery


class State(TypedDict, total=False):
    """整张图共享的状态。"""

    question: str
    plan: Plan

    # 不加 reducer：只有 search 一个节点写它，而 ranker 要的是覆盖不是追加。
    # reducer 是字段级的，加了就意味着任何节点写它都只能追加。
    papers: list[Paper]

    # 循环用的三个字段
    pending: list[SubQuery]   # 这一轮 search 要查的检索式（工作队列）
    tried: list[str]          # 已经试过的检索式，补检索时避开
    round: int                # 第几轮
    gaps: list[str]           # 证据不足的子问题（只有 check 写，不需要 reducer）

"""Search：把 pending 里的检索式跑掉。"""

import asyncio

import httpx

from ..sources import SOURCES
from ..state import State

TIMEOUT = 20


async def search(state: State) -> State:
    """结果**追加**到 papers，不是覆盖。

    循环第二轮进来时，第一轮筛出来的好文献要留着。papers 没有 reducer，
    所以这里显式地读旧值再拼。
    """
    pending = state["pending"]
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        batches = await asyncio.gather(*(
            src(client, sq) for sq in pending for src in SOURCES
        ))
    return {
        "papers": state.get("papers", []) + [p for b in batches for p in b],
        "tried": state.get("tried", []) + [sq.query for sq in pending],
        "pending": [],
    }

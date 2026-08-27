"""python -m paper_agent"""

import asyncio
import pathlib
import time

from .graph import graph

QUESTION = "单细胞 RNA-seq 的批次效应校正方法哪类更可靠"
OUT = pathlib.Path("report.md")


async def run(question: str) -> dict:
    """astream 一步一个补丁，顺便打出执行轨迹。"""
    state: dict = {}
    async for chunk in graph.astream({"question": question}):
        for node, patch in chunk.items():
            bits = [f"{k}={len(patch[k])}" for k in
                    ("papers", "evidence", "pending", "gaps") if k in patch]
            if "report" in patch:
                bits.append(f"report={len(patch['report'])}字")
            print(f"  · {node:9} {' '.join(bits)}")
            for k, v in patch.items():
                state[k] = state.get(k, []) + v if k == "evidence" else v
    return state


if __name__ == "__main__":
    t = time.perf_counter()
    out = asyncio.run(run(QUESTION))
    OUT.write_text(out["report"], encoding="utf-8")
    print(f"\n耗时 {time.perf_counter() - t:.1f}s  "
          f"证据 {len(out.get('evidence', []))} 条  ->  {OUT}")

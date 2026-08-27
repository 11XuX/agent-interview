"""python -m paper_agent"""

import asyncio
import time
from collections import Counter

from .graph import graph

QUESTION = "单细胞 RNA-seq 的批次效应校正方法哪类更可靠"

if __name__ == "__main__":
    async def run():
        state = {}
        async for chunk in graph.astream({"question": QUESTION}):
            for node, patch in chunk.items():
                bits = []
                if "pending" in patch:
                    bits.append(f"pending={len(patch['pending'])}")
                if "papers" in patch:
                    bits.append(f"papers={len(patch['papers'])}")
                if "gaps" in patch:
                    bits.append(f"gaps={len(patch['gaps'])}")
                if "round" in patch:
                    bits.append(f"round={patch['round']}")
                if node == "reader":
                    ok = sum(1 for x in patch["papers"] if x.sections)
                    chars = sum(len(s.text) for x in patch["papers"] for s in x.sections)
                    bits.append(f"全文 {ok}/{len(patch['papers'])} 篇 {chars} 字")
                print(f"  · {node:8} {' '.join(bits)}")
                state |= patch
        return state

    t = time.perf_counter()
    out = asyncio.run(run())

    for sq in out["plan"].sub_queries:
        print(f"\n■ {sq.question}")
        for p in out["papers"]:
            if sq.question in p.found_for:
                hits = f"x{len(p.found_for)}" if len(p.found_for) > 1 else "  "
                print(f"   {hits} [{p.source:9}] {p.year} {p.title[:52]}")

    print(f"\n共 {len(out['papers'])} 篇  "
          f"{dict(Counter(p.source for p in out['papers']))}  "
          f"耗时 {time.perf_counter() - t:.1f}s")

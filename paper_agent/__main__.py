"""python -m paper_agent"""

import asyncio
import time
from collections import Counter

from .graph import graph

QUESTION = "单细胞 RNA-seq 的批次效应校正方法哪类更可靠"

if __name__ == "__main__":
    t = time.perf_counter()
    out = asyncio.run(graph.ainvoke({"question": QUESTION}))

    for sq in out["plan"].sub_queries:
        print(f"\n■ {sq.question}")
        for p in out["papers"]:
            if sq.question in p.found_for:
                hits = f"x{len(p.found_for)}" if len(p.found_for) > 1 else "  "
                print(f"   {hits} [{p.source:9}] {p.year} {p.title[:52]}")

    print(f"\n共 {len(out['papers'])} 篇  "
          f"{dict(Counter(p.source for p in out['papers']))}  "
          f"耗时 {time.perf_counter() - t:.1f}s")

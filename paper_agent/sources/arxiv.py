"""arXiv。预印本，方法学论文常在这里先发。返回 Atom XML。"""

import xml.etree.ElementTree as ET

import httpx

from ..models import Paper, SubQuery

PER_QUERY = 3          # 每条检索式取几篇


async def fetch(client: httpx.AsyncClient, sq: SubQuery) -> list[Paper]:
    """arXiv。预印本，方法学论文常在这里先发。返回 Atom XML。

    ponytail: 官方 usage policy 要求 1 req/3s。一次运行只发几条，暂不加限流；
    真跑批量 eval 时要补。
    """
    ns = {"a": "http://www.w3.org/2005/Atom"}
    r = await client.get(
        "https://export.arxiv.org/api/query",      # 必须 https，http 返 301
        params={"search_query": f"all:{sq.query}", "max_results": PER_QUERY},
    )
    r.raise_for_status()
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        aid = (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
        out.append(Paper(
            source="arxiv",
            ext_id=aid,
            title=" ".join((e.findtext("a:title", "", ns) or "").split()),
            year=int((e.findtext("a:published", "", ns) or "0")[:4] or 0),
            abstract=" ".join((e.findtext("a:summary", "", ns) or "").split()),
            fulltext_url=f"https://arxiv.org/abs/{aid}",
            found_for=[sq.question],
        ))
    return out

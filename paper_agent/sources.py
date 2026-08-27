"""检索源。每个源一个 async 函数，签名统一。"""

import xml.etree.ElementTree as ET

import httpx

from .state import Paper, SubQuery

PER_QUERY = 3          # 每条检索式每个源取几篇


async def europepmc(client: httpx.AsyncClient, sq: SubQuery) -> list[Paper]:
    """Europe PMC。唯一能拿到带章节结构全文的源。"""
    r = await client.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": f"({sq.query}) AND OPEN_ACCESS:Y", "format": "json",
                "pageSize": PER_QUERY, "resultType": "core"},
    )
    r.raise_for_status()
    out = []
    for it in r.json().get("resultList", {}).get("result", []):
        pmcid = it.get("pmcid") or ""
        out.append(Paper(
            source="europepmc",
            ext_id=pmcid or it.get("id", ""),
            title=(it.get("title") or "").strip().rstrip("."),
            year=int(it.get("pubYear") or 0),
            abstract=it.get("abstractText") or "",
            fulltext_url=(f"https://www.ebi.ac.uk/europepmc/webservices/rest/"
                          f"{pmcid}/fullTextXML" if pmcid else ""),
            found_for=sq.question,
        ))
    return out


async def arxiv(client: httpx.AsyncClient, sq: SubQuery) -> list[Paper]:
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
            found_for=sq.question,
        ))
    return out


# OpenAlex 和 Semantic Scholar 暂时没接：
#   OpenAlex          走信用点配额（1000 点，每次扣 10），已被测试耗尽
#   Semantic Scholar  无 API key 时连发就 429
SOURCES = [europepmc, arxiv]

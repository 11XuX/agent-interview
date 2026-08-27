"""Europe PMC。唯一能拿到带章节结构全文的源。"""

import httpx

from ..models import Paper, SubQuery

PER_QUERY = 3          # 每条检索式取几篇


async def fetch(client: httpx.AsyncClient, sq: SubQuery) -> list[Paper]:
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
            found_for=[sq.question],
        ))
    return out

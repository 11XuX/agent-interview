"""业务对象。图里流转的都是这些，不是裸 dict。"""

import json
import re

from pydantic import BaseModel, Field, field_validator

class SubQuery(BaseModel):
    """一个子问题，以及查它该用的检索式。"""

    question: str = Field(description="子问题，中文，必须能被文献证据直接支撑或否定")
    query: str = Field(
        description=(
            "检索式，英文。核心概念之间用 AND，同义词用 OR 并括起来。"
            '例：(scRNA-seq OR "single-cell RNA-seq") AND "batch effect" AND (benchmark OR comparison)'
        )
    )

class Plan(BaseModel):
    """Planner 的产出。"""

    sub_queries: list[SubQuery] = Field(description="3-5 组，覆盖回答原问题所需的各个方面")

    @field_validator("sub_queries", mode="before")
    @classmethod
    def _parse_if_string(cls, v):
        """DeepSeek 偶尔把嵌套数组返成 JSON 字符串而不是数组，解一层。"""
        return json.loads(v) if isinstance(v, str) else v

class Relevance(BaseModel):
    """Ranker 对单篇文献的判断。"""

    score: int = Field(
        ge=0, le=3,
        description="0=完全无关；1=同领域但答不了这个问题；2=有部分证据；3=直接回答这个问题",
    )
    reason: str = Field(description="一句话说明依据，指出摘要里的哪一点")

class RetryQuery(BaseModel):
    """给证据不足的子问题换一条检索式。"""

    query: str = Field(description="换一个角度的检索式，英文，不要和已试过的雷同")

class Section(BaseModel):
    """全文里的一节。抽证据时要标出证据来自哪一节。"""

    title: str
    text: str


class Paper(BaseModel):
    """跨源统一后的一篇文献。"""

    source: str            # europepmc / arxiv
    ext_id: str            # PMCID / arXiv ID
    title: str
    year: int = 0
    abstract: str = ""
    fulltext_url: str = ""
    found_for: list[str] = []   # 命中了哪些子问题；被越多子问题命中说明越核心
    score: int = 0              # Ranker 打的相关性分，0-3
    reason: str = ""            # 打这个分的理由
    sections: list[Section] = []   # Reader 拉到的全文，已截断
    read_error: str = ""           # 拉全文失败的原因，空=没失败或没拉过

def _key(p: Paper) -> str:
    """去重键：标题规范化后精确匹配。

    ponytail: 跨源标题有细微差异（副标题、连字符、Unicode 破折号）时会漏判。
    真要收紧就上 token 集合的 Jaccard 相似度，但那要先有数据证明漏判率值得
    这个复杂度。
    """
    return re.sub(r"[^a-z0-9]+", "", p.title.lower())

def dedup(papers: list[Paper]) -> list[Paper]:
    """合并重复文献。

    不是丢掉重复项 —— 一篇被 4 个子问题同时命中，说明它是核心综述，
    这个信息要留下来。所以合并 found_for，保留先到的那条。
    """
    merged: dict[str, Paper] = {}
    for p in papers:
        k = _key(p)
        if k in merged:
            for q in p.found_for:
                if q not in merged[k].found_for:
                    merged[k].found_for.append(q)
        else:
            merged[k] = p.model_copy(deep=True)
    return list(merged.values())

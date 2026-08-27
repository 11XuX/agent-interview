"""数据结构。"""

import json
from typing import Annotated, TypedDict

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


class Paper(BaseModel):
    """跨源统一后的一篇文献。"""

    source: str            # europepmc / arxiv
    ext_id: str            # PMCID / arXiv ID
    title: str
    year: int = 0
    abstract: str = ""
    fulltext_url: str = ""
    found_for: str = ""    # 哪个子问题查出来的


class State(TypedDict, total=False):
    """整张图共享的状态。"""

    question: str
    plan: Plan
    papers: Annotated[list[Paper], list.__add__] #reducer函数

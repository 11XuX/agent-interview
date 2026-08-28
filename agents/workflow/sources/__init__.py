"""Paper Search Tools。每个源一个模块，都导出同签名的 fetch。

没接的两个源：
    OpenAlex          走信用点配额（1000 点，每次请求扣 10），已被测试耗尽
    Semantic Scholar  无 API key 时连发就 429
"""

from .arxiv import fetch as arxiv
from .europepmc import fetch as europepmc

SOURCES = [europepmc, arxiv]

__all__ = ["SOURCES", "europepmc", "arxiv"]

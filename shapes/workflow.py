"""形状一：写死的 workflow。就是 paper_agent/ 那张图。

8 个节点、2 条回边、1 次 Send 扇出，每条边都是代码写死的业务流程。
模型只在节点内部干活，从不决定"下一步做什么"。
"""

from paper_agent.graph import graph

__all__ = ["graph"]

"""形状三：deepagents harness。

比裸 ReAct 多出来的东西全是 middleware 白送的：

    文件系统   ls / read_file / write_file / edit_file / delete / glob / grep
    代码执行   execute
    子代理     task —— 委派给独立上下文窗口的子代理，跑完只带结果回来

对文献综述来说，文件系统的意义是**上下文卸载**：读到的全文写盘，
主对话里只留文件名，要用时再读回来。裸 ReAct 版没这个，读过的
全文会一直堆在 messages 里。
"""

from deepagents import create_deep_agent

from paper_agent.llm import llm

from .tools import TOOLS, read_section

SYSTEM = (
    "你是文献调研助手。用工具查资料，最后写一段带出处的综述。\n"
    "出处格式 [PMCxxxxx · 章节名]，只能引用你真的读到过的内容。\n"
    "查不够就继续查，不要急着下结论；材料够了就直接输出综述，不要再调工具。\n"
    "读到的长正文可以先 write_file 存盘再引用，别让正文一直占着对话。"
)

# 子代理：抽证据这件事上下文吃得多，交给独立窗口去做
SUBAGENTS = [{
    "name": "evidence-extractor",
    "description": "读一篇论文的指定章节，抽出能回答给定子问题的原文片段。"
                   "需要逐字引用原文时派它去，主对话不用背全文。",
    "system_prompt": "你只做一件事：从给定章节里找出能直接回答子问题的原文片段。"
                     "逐字引用，一个字不改。找不到就说找不到，不许编。",
    "tools": [read_section],   # 只给它读全文的能力，不给它检索
}]

graph = create_deep_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM, subagents=SUBAGENTS)

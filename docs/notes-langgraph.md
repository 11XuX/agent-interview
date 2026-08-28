# LangGraph 面试速记

基于 langgraph 1.2.11 / langchain-core 1.6.0 的实测。所有结论均在本项目
`paper_agent/`（workflow 形状）与 `shapes/`（三形状对照）中验证过。

---

## 一、执行模型

**本体是一个 while 循环** —— `pregel/main.py:2964` 的 `while loop.tick():`。
`tick()` 做三件事：检查步数上限、`prepare_next_tasks()` 算出这一轮该跑哪些节点、
没任务就返回 False。图只是一份"接下来跑谁"的数据，引擎在读它。

**与 agent loop 的唯一区别在谁决定下一步。** agent loop 由模型决定
（`stop_reason == "tool_use"`），LangGraph 由图结构决定（边）。LangGraph 是超集：
把"下一步跑谁"从 if-else 里抽出来变成可声明的数据；要模型驱动就用条件边把
`tool_calls` 接回去，`create_react_agent` 即此。

**超步（superstep）是 BSP 模型。** 一轮内所有活跃节点并发执行，全部完成后
统一提交状态更新。`_algo.py: apply_writes` 先按 `task_path` 排序保证确定性，
再按 channel 分组，最后在主循环里单线程依次应用 reducer。节点之间不共享可变
状态，无竞态、无需加锁。

**节点返回补丁，不是新状态。** 返回的 dict 只影响它提到的字段，其余原样保留。
LCEL 中同等效果需要 `RunnablePassthrough.assign` 一路携带。

**补丁里的未知字段被静默丢弃。** 返回 `{"conut": 999}` 不报错、不警告，值直接
消失。schema 加了字段但节点里拼错名、或节点写了 schema 未声明的字段，都表现为
下游取不到值而非异常。

---

## 二、reducer

**reducer 是并发写入的准入条件，不是语法糖。** 签名 `(旧值, 新值) -> 合并值`，
通过 `Annotated[类型, 函数]` 声明。

实测的四种组合：

| 场景 | 无 reducer | 有 reducer |
|:---|:---|:---|
| 串行（跨超步） | 后写覆盖 | 调用 `reducer(旧, 新)` 归约 |
| 并行（同超步） | `InvalidUpdateError: Can receive only one value per step` | 归约 |

并行场景下无 reducer 字段是**框架直接拒绝**，不是"后写的赢"。

**有 reducer 的字段被预初始化为单位元。** 实测：`Annotated[list, list.__add__]`
无人写过时读出 `[]`，`Annotated[int, int.__add__]` 读出 `0`，裸 `str` / 裸 `list`
则不在状态里。前者走 `BinaryOperatorAggregate` 通道需要归约起点，后者走 `LastValue`。
实际后果：节点里读带 reducer 的字段可直接下标，读裸字段必须 `.get()`。

**reducer 是字段级且全局的。** 一旦标记，任何节点写它都只能按该语义合并，
无法在某个节点选择覆盖。本项目曾给 `papers` 标 `list.__add__`，导致去重节点
返回 18 条被拼在原 19 条之后得到 37 条。解法是移除 reducer —— 只有一个节点
写它时本就不需要。

**判断标准**：只有一个节点写 → 不加；多个节点在同超步并发写 → 必须加；
循环中反复写且需累积 → 加。

**自定义 reducer 的实例**：跨源文献去重不能用 `add`，因为 Europe PMC 与
OpenAlex 收录同一篇是常态，重复会导致下游每篇多花一次全文抓取与一次抽取。
去重键必须是**一组键**而非单键 —— 有 DOI 用 DOI、否则用标题，会让同一篇论文
的有 DOI 版本与无 DOI 版本落入两个键空间而永不相撞。正确做法是返回
`{规范化标题, DOI}` 集合，任一命中即判为同篇。

---

## 三、边与控制流

**`add_edge(a, b)`** 是静态边。**`add_conditional_edges(源, 判断函数[, 映射表])`**
是动态边，判断函数签名 `(state) -> str`，返回下一个节点名或 `END`。

**判断函数不是节点** —— 不返回补丁、不能改状态，只回答"接下来跑谁"。

**映射表的作用是解耦**：判断函数返回业务标签（`"偶数"` / `"奇数"`），映射表
翻译成节点名。改图时只动映射表；且绘图时标签会显示在箭头上。

**回边构成循环** —— 条件边指向已执行过的节点。这是 LCEL 无法表达的形状：
`a | b | c` 在编译期定死步数，循环的轮数只有运行时才知道。

**`recursion_limit` 是兜底不是终止条件。** 默认 10007，可按次配置。撞上时抛
`GraphRecursionError`，**已完成的工作全部丢失**。业务上的步数上限必须自己在
状态里计数并由条件边返回 `END`，这样才能保留结果、写日志、给出降级输出。

---

## 四、Send 扇出

**唯一的动态扇出机制。** 条件边返回 `list[Send]` 而非节点名，框架在同一超步内
并发启动 N 个节点实例。扇出宽度由运行时状态决定。

```python
def fan_out(state) -> list[Send]:
    return [Send("extract", {"paper": p}) for p in state["papers"]]

builder.add_conditional_edges("reader", fan_out, ["extract"])
```

**被 Send 的节点收到的是载荷，不是图状态。** `Send("extract", {...})` 中的 dict
就是该实例看到的全部输入，它读不到 `state` 的其他字段，需要什么必须显式塞进载荷。

**Send 是 reducer 变成硬需求的场景。** N 个实例在同一超步返回同一字段，
不标 reducer 直接 `InvalidUpdateError`。

**超步天然是同步屏障。** `add_edge("extract", "synthesis")` 即可，所有 Send 实例
全部完成、状态合并完毕才进入下一节点，无需手写 join。

---

## 五、节点签名与依赖注入

**`StateNode` 是十种可接受签名的联合类型**（`graph/_node.py:76`）：

```
(state)                          最常用
(state, config)                  拿 RunnableConfig
(state, *, writer)               自定义流式输出
(state, *, store)                跨运行共享存储
(state, *, runtime)              运行时上下文
(state, *, config, writer, store)  以及各种组合
Runnable[NodeInputT, Any]        LCEL 链可直接作为节点
```

**按签名识别，无需注册。** 函数多声明一个 `config` 参数，框架就多注入一个。

**`add_node` 有两种形式**：`add_node("名字", 函数)`，或 `add_node(函数)` —— 后者
以函数名为节点名。

---

## 六、执行边界

实测（打印 pid/tid）：

| 节点类型 | 执行位置 |
|:---|:---|
| `async def` | 主线程主事件循环 |
| `def` | 同进程的线程池，pid 相同 tid 不同 |
| 任意 | **从不跨进程** |

**`invoke` 遇到 async 节点直接报错**：`TypeError: No synchronous function provided`。
反向可行 —— `ainvoke` 能跑同步节点。混合写法必须统一用 `ainvoke`。

**LangGraph 是单进程编排框架。** 所谓分布式能力全部来自状态可序列化：状态能存
能读，运行就能在任意进程续上；计算本身不分布。跨进程的单位是**整次运行**，
不是节点 —— 进程 A 跑到中断存档，进程 B 用同一 `thread_id` 恢复。

---

## 七、checkpointer 与 interrupt

**`interrupt()` 靠抛异常实现。** 内部抛 `GraphInterrupt`，框架捕获后存档并
**正常返回**给调用方，返回的 dict 多出 `__interrupt__` 键。不是把异常抛给调用方。

**恢复时节点从头重新执行。** 中断后 `get_state().next` 指向被中断的节点本身。
`Command(resume=x)` 之后该函数整体重跑，只是 `interrupt(...)` 直接返回 `x`。
**推论：interrupt 所在的节点必须是纯的、可重复执行的。** 把 interrupt 塞进含
LLM 调用的节点，每次恢复都会重复计费，且 temperature 非零时人确认过的内容与
实际执行的内容不一致。正确做法是拆出独立的确认节点。

**`thread_id` 是恢复的唯一钥匙。** 换 thread_id 即是全新运行。

**checkpointer 是 interrupt 的前提** —— 无存档无从恢复。`MemorySaver` 存进程
内存，进程结束即失效；跨进程需 `SqliteSaver` / `PostgresSaver`（独立的包）。

**状态里的自定义类必须登记 msgpack 白名单。** 未登记时当前版本仅警告，未来版本
会拒绝。实测严格白名单下未登记类**读回来变成 `dict`** —— 静默降级，程序不报错，
下游属性访问时才 `AttributeError`。

```python
JsonPlusSerializer(allowed_msgpack_modules=None).with_msgpack_allowlist([Paper, Plan, ...])
```

这是"状态必须可序列化"的代价：断点续跑与跨进程恢复建立在状态能存能读之上，
代价就是状态里不能放任意对象。

---

## 八、state / checkpointer / store 三层

| | 存什么 | 生命周期 | 跨进程 |
|:---|:---|:---|:---|
| state | 本次运行的数据 | 一次 invoke | 否 |
| checkpointer | state 的存档 | 按 `thread_id` | 换持久实现后可以 |
| store | 所有运行共享的数据 | 全局 | 换持久实现后可以 |

**去重放哪层取决于重复的范围**：一次运行内多源重复 → state；同用户多轮之间
重复 → checkpointer；所有运行之间重复 → store。

**`BaseStore` 不提供原子的 put-if-absent。** `if store.get(...) is None: store.put(...)`
是 check-then-act，多实例并发时会双写。强一致去重需绕过该接口直接用底层：
Postgres 的 `UNIQUE` 约束加 `ON CONFLICT DO NOTHING`，或 Redis 的 `SADD`。
`langgraph` 只自带 `InMemoryStore`，持久实现在独立的包中。

---

## 九、workflow / agent / harness 三种形状

Anthropic *Building effective agents* 的划分：workflow 是 LLM 与工具按预定义
代码路径编排，agent 是 LLM 动态决定自身流程与工具使用。

同一文献综述业务、同一模型、同一套工具，三道难度递增的题，实测均值：

| 形状 | 图规模 | 秒 | 模型调用 | 输入 token | 输出 token | 引用条数 |
|:---|:---|---:|---:|---:|---:|---:|
| workflow | 9 节点 2 回边 1 扇出 | 68 | 67 | 54200 | 8539 | 16.7 |
| ReAct | 2 节点 1 回边 | 94 | 10 | 47316 | 2653 | 12.3 |
| deepagents harness | 2 节点 + middleware | 97 | 9 | 68530 | 2661 | 13.7 |

**模型调用次数与 token 成本反相关。** workflow 调用 67 次却只花 54K 输入 token，
因为每次只喂该步所需材料；agent 调用 10 次花 47-68K，因为每一轮把整个对话历史
重发一遍。**agent 形状的 token 成本随轮次二次增长**，这是 harness 必须做上下文
管理（文件系统卸载、摘要压缩）的根本原因。

**分题结论完全反转。** 文献密集、术语标准的题上 workflow 产出 28 条引用而
agent 只有 10-12 条；交叉领域、同义词多的题上 workflow 撞上写死的
`MAX_ROUNDS=2` 上限，只产出 774 字、3 条引用，而 agent 查到满意为止得到 13-18 条。

**关键差异是失败模式**：workflow 的失败是**静默降级** —— 不报错，交出一份看似
正常实则敷衍的短报告；agent 的失败是**成本失控** —— 反复探索直到撞上
`recursion_limit`，且撞上时已完成的工作全部丢失。选型取决于更怕哪一种。

**判断标准**：步骤能否预先枚举。能枚举则 workflow，成本可估、失败可定位、
质量可用代码强制（例如引用真伪可做集合比对）；不能枚举才用 agent。

---

## 十、对接非 OpenAI 厂商的实测问题

**`with_structured_output` 默认 `method="json_schema"`**，DeepSeek 返回
`400 This response_format type is unavailable now`。改用 `method="function_calling"`
走 tool calling 通道。

**thinking 模式拒绝强制 tool_choice。** `function_calling` 必然下发
`tool_choice={"type":"function",...}`，与 DeepSeek V4 默认开启的思考模式冲突，
返回 `400 Thinking mode does not support this tool_choice`。解法是构造
`ChatOpenAI` 时 `extra_body={"thinking": {"type": "disabled"}}`。

**嵌套数组可能被返回为 JSON 字符串。** DeepSeek 偶发把 `list[SubQuery]` 字段
返成字符串，触发 `ValidationError: Input should be a valid list`。用前置校验器
兜住：

```python
@field_validator("sub_queries", mode="before")
@classmethod
def _parse_if_string(cls, v):
    return json.loads(v) if isinstance(v, str) else v
```

---

## 十一、工具设计

**`@tool` 编译成 JSON Schema，模型只能看到这份 schema**：函数名 → `name`，
docstring → `description`，参数名与类型标注 → `properties`，无默认值的参数 →
`required`。参数级描述用 `Annotated[类型, "说明"]` 或 `args_schema=PydanticModel`。

**工具报错不应抛异常。** 抛异常会终止整个 agent loop，模型连"这次没查到"都不知道。
返回错误文本则模型可见、可换个查法重试。错误信息要能指导下一步动作，例如
"检索失败 HTTP 400，检索式可能有语法问题，换一个试试"。

**失败隔离应放在最内层。** 每个数据源、每篇论文各自 try/except 并把失败原因写进
数据（`read_error` 字段），优于依赖 `gather(return_exceptions=True)` —— 后者只
知道少了一条，前者知道为什么少。

**`ToolNode` 只做执行，不做决策**：从最后一条消息读 `tool_calls`，按 name 查表
调用，把返回值包成 `ToolMessage`（带 `tool_call_id` 对齐）追加进 messages。

**MCP 解决的是分发问题不是能力问题。** 协议本体是 JSON-RPC 2.0，三条消息
（`initialize` / `notifications/initialized` / `tools/list`）即可拿到工具清单，
其结构与 `@tool` 的编译产物一致：`name` + `description` + `inputSchema`。
执行发生在服务端，客户端只传参收结果，因此权限边界由服务端界定。
三类原语的区别在控制方：tools 由模型决定调用，resources 由宿主应用决定是否
进入上下文，prompts 由用户主动触发。**单消费者场景上 MCP 是纯亏** —— 多一个
进程、多一次 IPC、工具列表从静态变运行时、多一层故障点，换来的复用无人使用。

---

## 十二、并发

**`asyncio.gather` 与 `Runnable.abatch` 的选用**：gather 用于任意协程，需自行
加 `Semaphore` 限并发；abatch 用于 Runnable 输入，`config={"max_concurrency": N}`
一个参数即可，且**返回顺序与输入顺序一致**，可直接 zip。

**限流器只管间隔，管不了总量。** 实测三种限流形态：

| 形态 | 表现 | 对策 |
|:---|:---|:---|
| 速率限制 | 两次请求需间隔 ≥ X 秒（arXiv 要求 1 req/3s） | 每源一个限流器 |
| 配额限制 | 总额度 1000 点、每次扣 10、约 16.5 小时重置（OpenAlex） | 限流器无效，需读响应头算总量 |
| 瞬时拒绝 | 服务端负载高时随机 429 | 退避重试 + `Retry-After` |

配额耗尽时返回的也是普通 429，与"太快了"无法从状态码区分，必须读
`x-ratelimit-remaining` 等响应头。

**限流器必须每源一个。** gpt-researcher 的 `GlobalRateLimiter` 是单例共用一个
delay，而实测各源速率要求相差一个数量级以上，共用阀门要么把快的拖慢十余倍、
要么把慢的打挂。

---

## 标准答法

**LangGraph 和 LangChain 什么关系。**
LCEL 用 `|` 组合 Runnable，得到编译期定死的有向无环管道，数据在算子之间穿过。
LangGraph 是节点读写共享状态、边描述跳转的状态机，支持环、支持中断恢复。
判断标准是流程形状：一条直线跑一次就用 LCEL，LangGraph 在那种场景是纯负担；
需要循环、需要多节点并发写同一累加器、需要中途停下来等人，才换 LangGraph。
两者不是替代关系 —— 节点内部的单步逻辑仍然用 LCEL 写最省事。

**reducer 是什么，什么时候需要。**
签名 `(旧值, 新值) -> 合并值` 的二元函数，通过 `Annotated` 声明在字段上。
它是并发写入的准入条件而非便利语法：同一超步内两个节点写同一个无 reducer 的
字段，框架直接抛 `InvalidUpdateError`，不是后写覆盖。判断标准是有几个节点写
这个字段 —— 只有一个就不要加，加了会挡住"覆盖"这种正常需求。真正需要它的
典型场景是 Send 扇出：N 个并发实例往同一个列表里写结果。

**human-in-the-loop 怎么做，恢复语义是什么。**
`interrupt()` 抛 `GraphInterrupt`，框架存档后正常返回，返回值里带 `__interrupt__`。
用 `Command(resume=值)` 加同一个 `thread_id` 恢复。关键是恢复时节点**从头重跑**
而非从中断行继续，所以 interrupt 必须放在纯节点里 —— 放进含 LLM 调用的节点会
重复计费，且人确认过的内容与实际执行的内容可能不一致。前提是 compile 时传了
checkpointer，且状态里的自定义类要登记 msgpack 白名单，否则严格模式下会静默
降级成 dict。

**什么时候用 workflow 什么时候用 agent。**
判断标准是步骤能否预先枚举。同一业务两种形状实测下来，workflow 快 30%、引用
密度高 36%，且质量可用代码强制校验；agent 在需要探索的题上能自适应，但 token
成本随轮次二次增长。真正的差异在失败模式：workflow 失败时静默降级，交出看似
正常实则敷衍的输出；agent 失败时成本失控，撞上递归上限且已完成的工作全丢。
所以能枚举就用 workflow，不能枚举才付 agent 的代价，而 harness 层（文件系统、
子代理、上下文压缩）解决的是 agent 形状 token 二次增长的问题。

# LangChain 面试速记

范围：LCEL 与 Runnable 协议。全部结论来自 `lc_demo.py` 六步实现的实测，模型为 DeepSeek V4 Flash，langchain-core 1.6。

## 一、Runnable 协议

- **唯一的抽象方法是 `invoke`** —— `Runnable.__abstractmethods__ == {'invoke'}`。基类定义 39 个公开方法，其余 38 个均带默认实现。
- **扩展方式** —— 继承 `Runnable` 并实现 `invoke`，自动获得 `batch`/`stream`/`ainvoke`/`abatch`/`astream`/`bind`/`with_retry`/`with_fallbacks`/`map`/`pick`/`assign`/`as_tool`。
- **`batch` 的默认实现** —— 线程池加循环调用 `self.invoke`。Template Method 模式：基类持有算法骨架，子类只提供变化的一步。
- **默认实现可被覆盖** —— `ChatOpenAI` 覆盖 `stream`（SSE 逐 token），`ChatPromptTemplate` 不覆盖，沿用基类实现（把 `invoke` 结果作为唯一 chunk 产出）。接口统一的代价是部分组件的部分能力是退化的。
- **继承树** —— `Runnable` → `RunnableSerializable`（增加 JSON 序列化，同时继承 `pydantic.BaseModel`）→ 各具体组件。`RunnableLambda` 直接继承 `Runnable`。
- **序列化的实际约束** —— 链中出现 `RunnableLambda` 则整条链不可序列化，LangServe 部署与 Hub 分享失效。

## 二、LCEL 的组合原语

组合方式只有两种：`a | b` 串行，`RunnableParallel(x=a, y=b)` 并行。

- **`__or__` 的实现** —— `return RunnableSequence(self, coerce_to_runnable(other))`。一行，无其他逻辑。
- **`coerce_to_runnable` 的四条规则** —— 已是 Runnable 则原样返回；生成器函数转 `RunnableGenerator`；callable 转 `RunnableLambda`；dict 转 `RunnableParallel`；其余抛 `TypeError`。所有隐式转换的唯一出处，包括 dict 内 value 位置上的裸函数（递归 coerce）。
- **`RunnableSequence.invoke`** —— 一个 for 循环，`input_ = step.invoke(input_)` 反复覆盖同一变量。不区分组件角色，因此 `prompt | llm | parser` 的顺序不是框架规定，只是这三者类型天然衔接。
- **展平规则** —— 串行相接时 steps 摊平合并；`RunnableParallel` 的分支不参与展平。`parse | match_chain` 得到 5 个 step 而非嵌套的 2 个。
- **`RunnableParallel` 是广播不是分发** —— 每个分支收到完整输入，各自取用所需字段，无需编写输入路由。
- **输出 key 由构造参数名决定**，与输入 key 无关。
- **`RunnableBranch` 是 LCEL 唯一的控制流原语** —— `(判断函数, 分支)` 依次匹配，末位参数为兜底分支。分支可以是任意 Runnable，包括不调用模型的纯函数。

## 三、结构化输出

- **`with_structured_output` 是打包不是合并** —— 返回 `RunnableSequence(_ChatModelBinding, PydanticToolsParser)`，等价于 `llm.bind(tools=[schema], tool_choice=指定) | PydanticToolsParser()`。三段式一段没少，只是后两段进了同一个盒子。
- **`.bind()` 的语义** —— 预先固定调用参数并返回新 Runnable，原对象不变。因此同一个 `llm` 实例可绑定不同 schema 供多条链并发使用。
- **Pydantic 类被编译进请求的 `tools` 字段** —— 类 docstring 成为 `function.description`，`Field(description=...)` 成为字段 description，`Field(ge=0, le=100)` 成为 `minimum`/`maximum`，`Literal[...]` 成为 `enum`。
- **两条独立的指令通道** —— `messages` 与 `tools` 是请求体的平级字段。前者承载任务描述，后者承载输出结构。字段填写不符预期改 `Field(description)`，任务方向不符预期改 prompt。
- **必填集合由 Pydantic 决定** —— 无 `Optional` 或默认值的字段全部进入 `required`。
- **嵌套模型被内联展开** —— LangChain 主动扁平化 `$ref`/`$defs`，因为部分厂商的 schema 解析器不支持引用。
- **三种 method 的机制差异**：

| method | 机制 | 保证强度 |
|---|---|---|
| `json_schema`（默认） | `response_format` 约束解码 | 采样层屏蔽非法 token，语法层保证 |
| `function_calling` | tools 参数生成 | 训练层保证，兼容性最好 |
| `json_mode` | 仅保证输出为合法 JSON | 最弱，schema 需写入 prompt |

- **借用 tool 通道的本质** —— 模型不执行任何工具，只输出调用意图。作为 schema 的 Pydantic 类是永不实现的假工具，取的是模型为调用它而生成的那份参数。由此继承该通道的全部限制。
- **模型只能产出 token** —— 三种 method 拿到的都是 JSON 文本，对象一律在客户端构造。差异仅在于 JSON 位于响应体的哪个字段，以及靠什么保证它合法。

## 四、类型契约

- **`|` 不做类型检查** —— 拼装成功不代表可执行。类型不匹配在 `invoke` 时于对应 step 抛出。
- **静态类型推断是尽力而为** —— 无注解的 lambda 推断为 `Any`，`chain.InputType`/`OutputType` 参考价值有限。
- **定位类型断点的手段** —— 逐步 invoke 打印实际类型：

```python
x = 输入
for step in chain.steps:
    y = step.invoke(x); print(type(x), '->', type(y)); x = y
```

- **标准三段的类型流** —— `dict` → `ChatPromptValue` → `AIMessage` → 目标类型。结构化输出时 `AIMessage.content` 为空字符串，数据位于 `tool_calls[0].args`，`finish_reason` 为 `tool_calls`。
- **对象与文本的往返** —— LLM 只接受文本，对象是给客户端代码用的。每跨越一次模型边界需解析进来与序列化出去各一次。喂给模型使用 `model_dump_json()`，不使用 `str()`（Pydantic 的 repr 格式不在模型的训练分布内）。
- **schema 即信息瓶颈** —— 解析阶段的模型未定义的字段永久丢失，下游无法恢复。结构化的代价是必须在解析前确定下游需要什么。

## 五、运行时能力

- **`batch` 保序** —— 并发执行，按输入顺序返回，可直接与输入 zip。需要按完成顺序取结果时使用 `batch_as_completed`。
- **`RunnableConfig` 逐层下传** —— `max_concurrency`、`callbacks`、`tags`、`metadata`、`recursion_limit`。每一步传递时挂接子 callback，LangSmith 的嵌套追踪树由此产生。
- **`max_concurrency` 是生产必需项** —— 低档 API key 的 QPS 上限通常为个位数，未限流的 `batch` 会触发大面积 429。
- **不可变装饰模式** —— `bind`、`with_retry`、`with_fallbacks`、`with_config` 一律返回新 Runnable，原链不变。
- **重试粒度** —— `chain.with_retry()` 失败时重跑整条链，含已成功的模型调用，产生无效计费。仅重试单步使用 `llm.with_retry()`。
- **流式能力取决于最不能流的算子** —— 实测同一条链仅替换末段 parser：`StrOutputParser` 产出 239 个 chunk，`PydanticToolsParser` 产出 1 个。后者必须收齐完整 JSON 才能构造对象。结构化输出与打字机效果互斥，需拆为两条链。
- **`astream_events`** —— 推送执行事件而非数据流，用于进度展示。

## 六、对接非 OpenAI 厂商的实测问题

以 DeepSeek V4 Flash 为例，三个错误按出现顺序：

1. **`400 This response_format type is unavailable now`** —— `with_structured_output` 默认 `method='json_schema'`，该厂商未实现 OpenAI Structured Outputs。改为 `method="function_calling"`。
2. **`400 Thinking mode does not support this tool_choice`** —— V4 系列默认开启 thinking；而 `function_calling` 为保证拿到结果，必然下发指定函数的 `tool_choice`。
3. **thinking 与工具调用的实际关系** —— 四种组合实测：

| thinking | tool_choice | 结果 |
|---|---|---|
| 开 | auto（不传） | 正常产生 tool_calls |
| 开 | `required` | 400 |
| 开 | 指定函数 | 400 |
| 关 | 指定函数 | 正常产生 tool_calls |

thinking 模式支持工具调用，不支持强制工具调用。强制 `tool_choice` 在解码层要求首 token 即进入工具调用语法，与"先产出 reasoning 再决定动作"的执行顺序冲突。Anthropic 的 extended thinking 有相同约束，`tool_choice` 仅接受 `auto` 与 `none`。

三种解法：

- `extra_body={"thinking": {"type": "disabled"}}` —— 关闭 thinking，保留结构化输出的确定性
- `disabled_params={"tool_choice": None}` —— 保留 thinking 并退回 auto，模型可能返回文本而不调用工具，需配合 `with_retry` 或 `include_raw=True` 判空
- `method="json_mode"` —— 不经过 tools 通道

选择依据为该步骤是否需要推理。抽取、分类、格式转换类任务关闭 thinking。

- **`extra_body` 的作用** —— 透传 OpenAI SDK 不识别、目标厂商识别的参数。厂商差异应收敛在此单点。
- **可移植的配置边界** —— 仅依赖 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`MODEL` 三个环境变量，Python 依赖仅 `langchain-openai`。

## 七、LCEL 的能力边界

- **拓扑在编译期确定** —— `chain = a | b | c` 构造完成后结构固定，运行时不能增删节点或改变走向。
- **组件无状态** —— `ChatOpenAI` 不保存对话历史，每次 invoke 为独立 HTTP 请求。多轮记忆需在链外显式维护并逐次传入。
- **`RunnableBranch` 是分支不是循环** —— 可以选择走哪条路，不能回到已执行过的节点。
- **单次 invoke 不可中断** —— 无法在中途暂停等待外部输入再恢复。
- **换用 LangGraph 的三个触发条件** —— 执行轮数由运行时状态决定；需要在节点间共享并累积状态；需要中途暂停等待人工输入。

## 标准答法

核心设计：

> LangChain 的核心是 Runnable 协议。唯一的抽象方法是 `invoke`，batch、stream、异步、重试、降级都是基类围绕 `invoke` 提供的默认实现。`|` 重载为 `RunnableSequence`，dict 隐式转 `RunnableParallel`，所以 LCEL 本质是类型化的函数组合。它的强项是声明式 DAG；拓扑在编译期定死，没有状态、没有循环、不能中途暂停，有状态的多轮流程需要 LangGraph。

输出格式稳定性：

> 三层机制。约束解码在采样层屏蔽非法 token，是唯一有语法保证的；tool calling 借用厂商为工具调用做的对齐训练，兼容性最好但属于概率保证；JSON mode 只保证是合法 JSON。Pydantic 校验是最后一道兜底，脏数据在 parser 层被拦下，不进入下游。

厂商兼容：

> OpenAI 兼容只覆盖 chat completions 的主干。`response_format` 的高级形态、`tool_choice` 的强制语义、thinking 参数各家实现不同。代码只依赖最小公共子集，厂商差异收敛到 `extra_body` 一处。

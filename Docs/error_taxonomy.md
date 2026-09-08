# 错误分类体系

这是 MedReason-Agent 的第一版错误分类。它不是最终版本，后续会根据真实实验错误样本继续修改。

## 当前自动归因标签

Phase 3 / Phase 4 当前使用下面六类 `Error Type`，写入 prediction record 的
`error_type` 和 `error_attribution` 字段：

| Error Type | 含义 |
| --- | --- |
| Perception Error | 图像看错，或 Vision Agent 没有给出可用观察。 |
| Retrieval Error | 证据检索错、证据缺失，或 Evidence Quality Score 过低。 |
| Reasoning Error | 图像和证据可用，但推理链或答案整合错误。 |
| Verification Error | Verifier 判断错误，例如支持了错误答案。 |
| Routing Error | Supervisor 选错策略，或实际 agent route 偏离期望路径。 |
| State Error | Agent 间信息传递错误，例如 shared state 缺失、压缩异常、claim status 丢失。 |

自动归因是启发式规则，不等同于最终医学事实判断。每条错误样本会记录：

```json
{
  "error_type": "Retrieval Error",
  "error_attribution": {
    "primary_error_type": "Retrieval Error",
    "candidate_error_types": ["Retrieval Error"],
    "signals": {
      "routing_error": false,
      "state_error": false,
      "retrieval_error": true,
      "verification_error": false,
      "perception_error": false
    },
    "requires_human_review": false
  }
}
```

当多个信号同时触发时，`primary_error_type` 使用优先级：

```text
Routing Error
State Error
Retrieval Error
Verification Error
Perception Error
Reasoning Error
```

这样设计是因为路由和状态错误会污染后续链路，应该优先归因。

## 旧版扩展错误类型

`VISUAL_PERCEPTION_ERROR`

视觉感知错误。模型看错图像、漏看图像中的明显发现，或把图像区域理解错。

`KNOWLEDGE_ERROR`

医学知识错误。模型缺少必要医学知识，或者错误使用医学知识。

`RETRIEVAL_ERROR`

检索错误。RAG 返回的 evidence 不相关、误导、不充分，或者没有检索到关键知识。

`REASONING_ERROR`

推理错误。模型已经有足够信息，但推理链条出错，最终得到错误答案。

`LOGICAL_LEAP`

逻辑跳跃。结论超出了图像、问题或 evidence 能支持的范围。

`HALLUCINATION`

幻觉。回答中编造了输入里不存在的图像发现、患者信息、医学证据或上下文。

`SUPERVISOR_ROUTING_ERROR`

Supervisor 路由错误。比如知识密集问题本该检索 evidence，但 Supervisor 选择了 Direct 路径。

`TOOL_SELECTION_ERROR`

工具选择错误。系统选择了错误工具，或在错误时间调用了工具。

`VERIFICATION_ERROR`

验证错误。Verifier 没有发现错误，或者错误地否定了本来正确的答案。

`CORRECT_TO_WRONG_REVISION`

正确答案被改错。初始答案正确，但 Critic / Revision 步骤把它改成了错误答案。

`FINAL_ANSWER_FORMAT_ERROR`

最终答案格式错误。语义上可能有答案，但没有遵守要求的输出格式，导致评估脚本无法正确读取。

`UNKNOWN`

未知错误。无法可靠判断错误来源时使用。

## 错误分析规则

每次完整实验结束后，都要抽样 incorrect predictions 并分类。不能只靠感觉修改系统。

示例：

```text
100 个错误样本：
35% visual perception
30% reasoning
15% retrieval
10% supervisor routing
10% other
```

## Critic 的特殊风险

Verification Agent 不是天然有益的。它可能：

- 把错误答案改对。
- 发现 hallucination。
- 也可能把正确答案改错。

因此评估 Critic 时必须同时报告：

- Correction Rate
- Regression Rate / Correct-to-Wrong Revision Rate

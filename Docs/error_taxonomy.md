# 错误分类体系

这是 MedReason-Agent 的第一版错误分类。它不是最终版本，后续会根据真实实验错误样本继续修改。

## 错误类型

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

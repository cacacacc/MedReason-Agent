# 研究问题

本文档把 MedReason-Agent 的项目想法改写成可以通过实验回答的研究问题。后续所有代码、实验和报告都要服务于这些问题，而不是只做一个看起来复杂的 Demo。

## 总问题

在固定同一个视觉语言模型 backbone 的前提下，Supervisor-guided Multi-Agent 推理框架是否能比 Direct VLM inference 更可靠地完成医学多模态视觉问答？

这里的“更可靠”不只指准确率更高，还包括更少幻觉、更好的证据支持、更合理的置信度、更低的 token / latency 成本。

## RQ1：显式推理是否有帮助

问题：

显式推理，例如 Chain-of-Thought，是否能提升医学多模态问答表现？

实验做法：

- 比较 Direct VLM、Structured Direct Prompt、Chain-of-Thought。
- 固定同一个模型、同一个数据划分、同一套图像预处理、同一套解码参数。

主要指标：

- Accuracy / Exact Match
- 开放式答案归一化后的匹配结果

辅助指标：

- 推理一致性
- 输出 token 数
- latency

## RQ2：多 Agent 协作是否有帮助

问题：

Multi-Agent collaboration 是否比单个 VLM 或简单 CoT 更适合医学多模态推理？

实验做法：

- 比较 Direct VLM、CoT、Fixed Multi-Agent。

关键风险：

- Agent 越多不一定越好。更多 Agent 可能引入更多中间错误、更长上下文、更高成本，以及互相不一致的中间结论。

## RQ3：Supervisor 路由是否优于固定流程

问题：

Supervisor Agent 能否根据问题难度和任务类型动态选择推理路径，并优于固定 Multi-Agent pipeline？

实验做法：

- 比较 Fixed Multi-Agent 和 Supervisor Multi-Agent。
- 尽量保持可用 Agent、模型 backbone、提示词版本和评估脚本一致。

指标：

- Accuracy
- 平均 Agent 调用次数
- 平均推理步数
- latency
- token usage

## RQ4：Verification / Critic 是否能减少错误

问题：

Verification Agent 或 Critic Agent 是否能减少 hallucination，并修正错误推理？

实验做法：

- 比较 Supervisor without Critic 和 Supervisor with Critic。
- 同时记录“纠错”和“改错”两种情况。

指标：

- Error Correction Rate
- Correct-to-Wrong Revision Rate
- Unsupported Claim Rate
- Hallucination Rate

## RQ5：自适应推理是否改善准确率与成本平衡

问题：

Adaptive Reasoning 能否让简单问题走短路径、困难问题走完整推理路径，从而在准确率和效率之间取得更好平衡？

实验做法：

- 比较 Always Full Reasoning 和 Adaptive Routing。

指标：

- Accuracy
- token usage
- latency
- cost per task

## RQ6：置信度是否可靠

问题：

模型输出的 confidence 是否经过良好校准？Verification 是否能改善校准？

实验做法：

- 收集每种方法的 confidence。
- 比较 confidence 和实际 correct / incorrect 之间的关系。

指标：

- Expected Calibration Error, ECE
- Brier Score
- Reliability Diagram

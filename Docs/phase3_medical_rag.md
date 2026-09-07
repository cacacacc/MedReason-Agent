# Phase 3：Medical RAG

## Phase Goal

Phase 3 研究外部医学证据是否能提升医学多模态 VQA 的可靠性。

Phase 3 现在包含两条可运行的 RAG agent 路径。
在这两条路径之外，本项目还加入第三条组合路径，用来研究 RAG 和 Verifier 如何配合。

Knowledge RAG 目标架构：

```text
Question -> Retriever -> Reranker -> Evidence Filter -> VLM Reasoning -> Answer
```

Evidence RAG 目标架构：

```text
Image + Question -> Reasoning Agent -> Claim
Claim -> Evidence Agent -> Retriever -> Reranker -> Evidence Filter
Claim + Evidence -> Verifier Agent -> Verified Answer
```

Knowledge + Evidence RAG Verifier 目标架构：

```text
Knowledge Agent
-> Retrieve knowledge
-> Reasoning Agent
-> Generate Claims
-> Verifier Extracts Claims
-> Evidence Agent retrieves support
-> Verifier compares Claim vs Evidence
-> SUPPORTED / UNSUPPORTED / CONTRADICTED
-> Final Answer
```

核心比较：

```text
CoT without RAG vs RAG-Augmented Reasoning
Knowledge RAG vs Evidence RAG
Knowledge-only RAG vs Knowledge + Evidence Verification
```

## Why

Direct VLM 和 CoT 主要依赖模型内部知识与图像理解。Medical RAG 会在生成答案前先检索医学 evidence，然后把 evidence 放进 prompt。

RAG 可能帮助模型减少 hallucination，也可能因为检索到无关 evidence 而让答案变差。所以 Phase 3 必须同时评估两件事：

- 最终答案是否更准确。
- 检索到的 evidence 本身是否高质量。

本阶段已经实现两种 RAG agent：

- **KnowledgeRAGAgent**：reasoning 之前检索知识，回答“先查知识是否能帮助模型推理”。
- **EvidenceRAGAgent**：reasoning 之后检索证据，回答“模型已经提出的 claim 是否被证据支持”。
- **KnowledgeThenEvidenceRAGAgent**：先用 Knowledge RAG 帮助生成 claims，再用 Evidence RAG + Verifier 检查 claims 是否被支持。

## Theory

`RAG`

RAG 是 Retrieval-Augmented Generation，中文可以理解为“检索增强生成”。模型不是只靠内部参数回答，而是先查找相关资料，再结合资料回答。

本项目例子：如果问题问胸片里是否有 opacity，retriever 可能返回关于 pulmonary opacity / consolidation 的医学摘要片段，VLM 再结合图像和 evidence 回答。

为什么需要它：医学问题经常需要外部知识。RAG 可以测试 external evidence 是否能让 frozen VLM 的推理更可靠。

`Knowledge RAG`

Knowledge RAG 是“先查知识，再开始推理”。它发生在 Reasoning 之前。

通俗解释：我还不知道某个图像 finding 可能代表什么，所以先查资料，再做判断。

本项目例子：

```text
Finding: right lung opacity
-> Knowledge Agent retrieve: possible causes of right lung opacity
-> Clinical Reasoning Agent uses retrieved knowledge to answer
```

所以：

```text
Knowledge RAG = Knowledge Acquisition
```

当前 `exp03_rag.yaml` 的默认 baseline 属于 Knowledge RAG：

```text
Question -> Retriever -> Reranker -> Evidence Filter -> VLM Reasoning -> Answer
```

实验记录中的 route 是：

```text
knowledge_agent -> retriever -> reranker -> evidence_filter -> clinical_reasoning_agent
```

`Evidence RAG`

Evidence RAG 是“先有推理结论，再查证据验证”。它发生在 Reasoning 之后，通常属于 Verification 阶段。

通俗解释：我已经提出一个 claim，现在要查证据，看这个 claim 是否站得住。

本项目例子：

```text
Claim: This finding is compatible with pneumonia.
-> Evidence Agent retrieve: evidence supporting pneumonia for right lower lung opacity
-> Verifier compares Claim vs Retrieved Evidence
```

所以：

```text
Evidence RAG = Claim Verification
```

核心区别：

```text
Knowledge RAG: 先查再推理
Evidence RAG: 先推理，再查证据验证
```

这两种 RAG 可以同时存在，但必须在实验记录中分开标记，否则会混淆“知识增强”和“结论验证”的效果。

当前 Evidence RAG 的实验配置是：

```text
configs/experiments/exp03_evidence_rag.yaml
```

实验记录中的 route 是：

```text
reasoning_agent -> claim_extractor -> evidence_agent -> retriever -> reranker -> evidence_filter -> verifier_agent
```

Evidence RAG 会额外记录：

```text
initial_prediction
initial_reasoning_output
verified_claim
evidence_query
```

其中 `evidence_query` 来自初始 claim，而不是只来自原始 question。

`Knowledge + Evidence RAG Verifier`

组合路径用于研究两种 RAG 是否能互补：

```text
Knowledge RAG: 给 Reasoning Agent 提供背景知识
Evidence RAG: 给 Verifier 提供 claim-level 支持证据
```

完整执行顺序是：

```text
Question
-> Knowledge Agent retrieves pre-reasoning knowledge
-> Reasoning Agent generates answer and claims
-> Claim Extractor selects checkable claim
-> Evidence Agent retrieves support / contradiction evidence
-> Verifier compares claim vs evidence
-> SUPPORTED / UNSUPPORTED / CONTRADICTED
-> Final Answer
```

对应实验配置：

```text
configs/experiments/exp03_knowledge_evidence_rag.yaml
```

对应结果字段：

```text
rag_mode: knowledge_then_claim_verification
knowledge_query: 原始问题
evidence_query: 基于 claim 构造的验证 query
generated_claims: Reasoning Agent 生成的可验证 claims
claim_verification_status: SUPPORTED / UNSUPPORTED / CONTRADICTED
critic_decision: 同 claim_verification_status
```

`Retriever`

Retriever 是检索器，负责从医学知识库里找出和问题最相关的 chunks。

当前实现：`KeywordRetriever`，轻量、确定性、适合先跑通实验闭环。

目标实现：BGE embedding + Qdrant vector retrieval。

`Reranker`

Reranker 是重排器。Retriever 先粗召回候选证据，reranker 再根据问题和 evidence 的相关性重新排序。

当前实现：`KeywordReranker`，使用问题词和 evidence 标题/正文的重叠度打分。

`Evidence Filter`

Evidence Filter 负责过滤低质量 evidence，只把最终 top-k evidence 交给 VLM。这样可以减少 prompt 长度，也减少无关证据误导模型的风险。

`Evidence Quality Score`

Evidence Quality Score 是本阶段新增的证据质量指标。它不直接评价最终答案是否正确，而是评价 retrieved evidence 是否与当前问题和标准答案相关。

当前规则版分数范围是 0 到 1：

```text
question_coverage = evidence 覆盖了多少问题内容词
answer_coverage = evidence 覆盖了多少开放式标准答案内容词
```

如果 ground truth 是 `yes/no` 这类 closed answer，`answer_coverage` 不参与总分，因为 evidence 里出现 yes/no 并不能说明证据质量高。

开放式答案：

```text
evidence_quality_score = 0.6 * question_coverage + 0.4 * answer_coverage
```

Closed answer：

```text
evidence_quality_score = question_coverage
```

为什么需要它：如果 RAG 准确率下降，我们要知道是检索证据差，还是模型没有正确使用证据。Evidence Quality Score 帮助区分这两种错误来源。

`Structured RAG Prompt`

Structured RAG Prompt 是结构化 RAG 提示词。它不是简单写一句 “Use the context to answer”，而是明确约束模型如何使用图像和证据。

本项目的 RAG prompt 至少约束四件事：

```text
1. Use retrieved evidence explicitly.
2. Do not introduce unsupported medical claims.
3. Separate image observation from external knowledge.
4. State uncertainty if evidence is insufficient.
```

当前 `rag_v3` 输出结构：

```text
Observation:
Retrieved Evidence:
Reasoning:
Unsupported Assumptions:
Uncertainty:
Conclusion:
Final Answer:
```

为什么需要它：RAG 的风险不是“有 evidence 就一定更可靠”。如果模型把图像观察、外部知识和自己猜测混在一起，我们就很难做 error analysis。因此 prompt 必须强制模型分开写：哪些来自 image，哪些来自 retrieved evidence，哪些是假设，最终答案是什么。

## 训练与配置调整

Phase 3 之前的训练配置不需要改成“训练模型”，因为本项目主实验仍然坚持：

```text
Frozen Backbone
training_epochs: 0
```

Knowledge RAG 使用下面的 prompt 配置：

```text
prompt_version: rag_v3
prompt_contract: structured_rag_v3
rag_mode: knowledge_acquisition
```

Evidence RAG 使用下面的 prompt 配置：

```text
prompt_version: evidence_rag_v1
prompt_contract: structured_evidence_verification_v1
rag_mode: claim_verification
```

Knowledge + Evidence RAG Verifier 使用下面的 prompt 配置：

```text
prompt_version: knowledge_evidence_rag_v1
prompt_contract: structured_knowledge_then_evidence_verification_v1
rag_mode: knowledge_then_claim_verification
```

原因是两者研究的问题不同：Knowledge RAG 测试知识获取，Evidence RAG 测试 claim verification。它们不能混在同一个实验编号或同一份结果里。

## Files

```text
CREATE
src/medreason_agent/agents/__init__.py
src/medreason_agent/agents/rag_agents.py
src/medreason_agent/evaluation/evidence_metrics.py
tests/test_rag_agents.py
tests/test_evidence_metrics.py
configs/experiments/exp03_evidence_rag.yaml
configs/experiments/exp03_knowledge_evidence_rag.yaml

MODIFY
Docs/phase3_medical_rag.md
Docs/experiment_protocol.md
configs/base.yaml
scripts/run_rag.py

DELETE
无
```

## Current Implementation Scope

当前 Phase 3 是可运行的双路径 RAG scaffold：

- 构建小型 seed medical knowledge base。
- 把医学文档切成 chunks。
- 用 keyword retriever 召回候选 evidence。
- 用 keyword reranker 重排候选 evidence。
- 用 evidence filter 保留最终 evidence。
- 使用 `KnowledgeRAGAgent` 执行“先查再推理”。
- 使用 `EvidenceRAGAgent` 执行“先推理 claim，再查证据验证”。
- 使用 `KnowledgeThenEvidenceRAGAgent` 执行“先知识增强，再 claim-level verification”。
- 在结果中记录 `rag_mode`、`agent_route` 和 `evidence_query`。
- 保存 `retrieved_evidence`、`evidence_quality_score`、`reasoning_output` 和答案指标。

这还不是最终 PubMed/BGE/Qdrant RAG 系统。

## Outputs

RAG prediction record 现在会额外包含：

```json
{
  "rag_mode": "knowledge_acquisition",
  "evidence_query": "Is there right lung opacity?",
  "initial_prediction": "",
  "verified_claim": "",
  "evidence_quality_score": 0.75,
  "evidence_quality": {
    "evidence_quality_score": 0.75,
    "question_coverage": 0.5,
    "answer_coverage": 1.0,
    "num_evidence": 3,
    "has_evidence": true
  }
}
```

Evidence RAG record 示例：

```json
{
  "rag_mode": "claim_verification",
  "evidence_query": "medical evidence supporting or contradicting this claim...",
  "initial_prediction": "pneumonia",
  "verified_claim": "pneumonia",
  "agent_route": [
    "reasoning_agent",
    "claim_extractor",
    "evidence_agent",
    "retriever",
    "reranker",
    "evidence_filter",
    "verifier_agent"
  ]
}
```

组合 RAG Verifier record 示例：

```json
{
  "rag_mode": "knowledge_then_claim_verification",
  "knowledge_query": "Is there right lung opacity?",
  "evidence_query": "medical evidence supporting or contradicting this claim...",
  "generated_claims": ["right lung opacity is present"],
  "claim_verification_status": "SUPPORTED",
  "critic_decision": "SUPPORTED",
  "agent_route": [
    "knowledge_agent",
    "retriever",
    "reranker",
    "evidence_filter",
    "reasoning_agent",
    "claim_extractor",
    "evidence_agent",
    "retriever",
    "reranker",
    "evidence_filter",
    "verifier_agent"
  ]
}
```

RAG `metrics.json` 现在会额外包含：

```json
{
  "mean_evidence_quality_score": 0.42,
  "mean_question_coverage": 0.38,
  "mean_answer_coverage": 0.25,
  "evidence_empty_rate": 0.1
}
```

## Dataset Scale

VQA-RAD 实验规模遵守全局策略：

```text
Smoke: 1-20 test samples
Dev: 100 test samples
Full: 451 test samples
```

CPU 上跑真实 Qwen 时先从 1 条样本开始。

## Pass Criteria

- `scripts/build_medical_kb.py --seed-medical-vqa` 能生成本地 chunk 文件。
- `scripts/run_rag.py` 能写出 predictions 和 metrics。
- 每条 prediction record 包含 `retrieved_evidence`。
- Knowledge RAG record 的 `agent_route` 从 `knowledge_agent` 开始。
- Evidence RAG record 的 `agent_route` 包含 `reasoning_agent`、`claim_extractor`、`verifier_agent`。
- Evidence RAG record 包含 `initial_prediction`、`verified_claim` 和基于 claim 的 `evidence_query`。
- 组合路径 record 包含 `knowledge_evidence` 和 `verification_evidence` 两类 evidence。
- 组合路径的 verifier 必须输出 `SUPPORTED / UNSUPPORTED / CONTRADICTED`。
- 每条 RAG record 包含 `evidence_quality_score`。
- `metrics.json` 包含 `mean_evidence_quality_score`。
- `reasoning_output` 保留完整 RAG reasoning text。
- `prediction` 能优先从 `Final Answer:` 抽取，也能从 `Conclusion:` 兜底抽取。
- pytest 和 ruff 全部通过。

## Research Questions

1. 为什么 RAG 可能减少 hallucination？
2. 为什么无关 evidence 可能让答案变差？
3. 为什么必须为每条 prediction 保存 retrieved evidence？
4. 为什么 Evidence Quality Score 不能替代最终 Accuracy？
5. 为什么 keyword retrieval 不能被当成最终 RAG 方法？
6. 为什么 Knowledge RAG 和 Evidence RAG 需要分开记录？

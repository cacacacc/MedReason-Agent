# Accuracy Recovery Plan

后续阶段 accuracy 下降的主要原因不是模型训练失败，而是复杂 pipeline 与 VQA-RAD
短答案评估目标不匹配：

```text
CoT 增加 reasoning drift
RAG 引入通用医学知识噪声
Verifier 过度保守
Agent 链路带来误差传播
Final Answer 过长导致 exact match 吃亏
```

## 已实现优化

### 1. 强制短答案

所有 Direct / CoT / RAG / Multi-Agent prompt 都加入：

```text
For yes/no questions, answer exactly yes or no.
For open questions, answer with one short phrase.
```

### 2. 统一答案规整

新增：

```text
src/medreason_agent/answer_normalization.py
```

作用：

```text
yes/no 问题中，"No, ..." -> "no"
开放题只保留第一行短答案
保留 raw_output 用于人工复核
```

### 3. 更强 answer normalization

`exact_match` 现在支持：

```text
question-aware yes/no canonicalization
常见医学缩写归一化，例如 cerebrospinal fluid -> csf
```

### 4. Simple Visual Question 跳过 RAG

新增 `question_routing: heuristic`。

规则：

```text
CLOSED + PRES / MODALITY / PLANE / ORGAN / COUNT / COLOR
-> Vision Agent + Answer Agent

POS / SIZE / ATTRIB
-> Vision Agent + Reasoning Agent + Answer Agent

ABN / OTHER / diagnosis / disease / cause / abnormality
-> Vision Agent + Retrieval Agent + Reasoning Agent + Verifier Agent + Answer Agent
```

这个开关只在新配置中启用，不改变旧实验结果。

## 新增调参配置

Phase 4：

```text
configs/experiments/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_100.yaml
```

Phase 5：

```text
configs/experiments/exp05_separate_verifier_heuristic_routing_qwen_7b_4090d_pmc10k_100.yaml
```

## 推荐验证顺序

先跑原始 supervisor 和 heuristic supervisor：

```bash
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_100.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_100.yaml
```

再跑 Phase 5：

```bash
python scripts/run_phase5_verification.py --config configs/experiments/exp05_separate_verifier_qwen_7b_4090d_pmc10k_100.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_separate_verifier_heuristic_routing_qwen_7b_4090d_pmc10k_100.yaml
```

如果 heuristic routing 提升 accuracy，同时降低 retrieval/verifier calls 和 latency，
说明主要问题是“简单题被复杂工具干扰”。后续 full 主实验应优先使用 heuristic routing
作为 Supervisor 的可解释路由约束。

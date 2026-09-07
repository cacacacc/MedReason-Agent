# Phase 2：Chain-of-Thought Reasoning

## Phase Goal

本阶段研究显式推理是否能提升医学多模态问答。实验形式是：

```text
Image + Question -> VLM -> Reasoning -> Final Answer
```

核心比较：

```text
Direct Answer vs CoT Prompt
```

## Why

Direct VLM 直接输出答案，速度快，但我们看不到模型为什么这样回答。CoT 让模型先写出推理过程，再输出最终答案。这样可能帮助模型处理复杂问题，也可能因为写了错误推理而把答案带偏。

因此 Phase 2 不能只问“CoT 准确率是否更高”，还要问：

- 哪些问题 CoT 改对了？
- 哪些问题 CoT 改错了？
- CoT 的额外 token 和 latency 是否值得？

## Theory

`Chain-of-Thought`

Chain-of-Thought，简称 CoT，意思是“思维链”。它让模型在给最终答案前先生成中间推理步骤。

通俗解释：不是让学生直接写答案，而是要求他写出解题过程。

本项目例子：模型先描述图像观察，再说明如何从观察推到答案，最后输出 `Final Answer`。

为什么需要它：医学 VQA 中有些问题不是简单识别物体，而是需要把图像线索和医学知识连接起来。CoT 可能帮助模型显式组织这些线索。

## Files

```text
CREATE
Docs/phase2_cot_reasoning.md
src/medreason_agent/prompts/cot.py
src/medreason_agent/analysis/__init__.py
src/medreason_agent/analysis/compare_predictions.py
scripts/run_cot.py
scripts/compare_direct_vs_cot.py
configs/experiments/exp02_cot.yaml
configs/experiments/exp02_cot_qwen_smoke.yaml
tests/test_cot_prompt.py
tests/test_compare_predictions.py

MODIFY
src/medreason_agent/models/vlm.py
README.md

DELETE
无
```

## Tasks

1. 新增 CoT prompt。
2. 让 VLM 输出 `Observation / Reasoning / Final Answer`。
3. 从模型完整输出中提取最终答案。
4. 保存完整 reasoning output。
5. 计算 Accuracy / Token F1 / BLEU-1。
6. 对比 Direct 和 CoT 的逐题结果。

## Experiments

Smoke test：

```text
exp02_cot
backend: mock
split: test
max_samples: 20
```

真实 Qwen CPU smoke：

```text
exp02_cot_qwen_smoke
backend: qwen2_5_vl
split: test
max_samples: 1
```

## Dataset Scale

```text
Smoke Test: 20 samples
Dev Experiment: 100-200 samples
Full Experiment: 451 test samples
```

CPU 上跑真实 Qwen 时先从 1 条样本开始。

## Pass Criteria

- CoT runner 能生成 `predictions.jsonl` 和 `metrics.json`。
- 每条记录保留完整 `reasoning_output`。
- 能从 `Final Answer:` 中提取用于评估的短答案。
- Direct vs CoT comparison 能输出 helped / hurt / both correct / both wrong。
- pytest 和 ruff 全部通过。

## Research Questions

1. 为什么 CoT 可能提升复杂问题表现？
2. 为什么 CoT 也可能降低准确率？
3. 为什么我们必须单独保存 `reasoning_output`，而不是只保存最终答案？

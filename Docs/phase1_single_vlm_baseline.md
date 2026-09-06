# Phase 1：Single VLM Baseline

## Phase Goal

建立第一个可复现的 Single VLM baseline。这个 baseline 的形式很简单：

```text
Image + Question -> VLM -> Answer
```

这个阶段的重点不是追求最高分，而是建立后续所有方法都要比较的参照点。

## Why

如果没有 Direct VLM baseline，后面就无法判断 CoT、RAG、Fixed Multi-Agent、Supervisor Multi-Agent 是否真的有效。

Baseline 的作用是回答：

> 不加任何复杂 reasoning architecture 时，同一个 VLM 本身能做到什么水平？

## Theory

本阶段需要理解三个概念：

`Baseline`

Baseline 是对照系统。它提供最低限度但公平的参照，用来判断新方法是否真的带来提升。

`Direct VLM Inference`

Direct VLM inference 是直接把图像和问题输入视觉语言模型，让模型直接输出答案，不要求它显式写推理过程。

`Controlled Comparison`

Controlled comparison 是受控对比。比较两个方法时，除了被研究的变量，其他条件都要尽量相同。

## Files

本阶段新增或修改：

```text
CREATE
Docs/phase1_single_vlm_baseline.md
src/medreason_agent/data/vqa_rad.py
src/medreason_agent/evaluation/answer_metrics.py
src/medreason_agent/models/vlm.py
src/medreason_agent/prompts/direct.py
scripts/run_direct_vlm.py
tests/test_answer_metrics.py
tests/test_vqa_rad_loader.py
tests/test_run_direct_vlm.py

MODIFY
configs/experiments/exp01_direct_vlm.yaml
README.md

DELETE
无
```

## Tasks

1. 读取 `Data/Processed/vqa_rad/test.jsonl`。
2. 生成 Direct VLM prompt。
3. 调用 VLM backend。
4. 保存每条 prediction record。
5. 计算 baseline metrics。
6. 输出 `predictions.jsonl` 和 `metrics.json`。

## Experiments

当前先跑 smoke test：

```text
exp01_direct_vlm
split: test
max_samples: 20
backend: mock
```

`mock` backend 只用于验证 pipeline，不代表真实模型结果。真实 baseline 需要切换到 Qwen2.5-VL backend 后重新运行。

## Dataset Scale

当前 VQA-RAD OSF 公共版实际导入统计：

```text
all: 2248
train: 1527
validation: 270
test: 451
```

Phase 1 的规模策略：

```text
Smoke Test: 20 test samples
Dev Experiment: 100-200 test samples
Full Experiment: 451 test samples
```

## Pass Criteria

Smoke test 完成标准：

- 能读取 VQA-RAD processed split。
- 能生成每条预测记录。
- 能保存 JSONL 和 JSON metrics。
- 测试和 lint 通过。

Full baseline 完成标准：

- 使用真实 Qwen2.5-VL backend。
- 跑完整 451 条 test samples。
- 保存完整 predictions 和 metrics。
- README / report 中明确记录模型、prompt、数据版本、解码参数和运行限制。

## Research Questions

1. 为什么 Direct VLM baseline 是后续所有复杂方法的参照点？
2. 为什么 `mock` backend 的结果不能写进论文主结果表？
3. 为什么 Single VLM baseline 阶段不应该加入 CoT、RAG 或 Verifier？

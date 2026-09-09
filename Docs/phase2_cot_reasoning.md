# Phase 2：Reasoning Baseline

## Phase Goal

本阶段研究显式推理是否能提升医学多模态问答。

核心问题：

```text
Reasoning 有没有帮助？
结构化 reasoning 是否比普通 step-by-step 更稳定？
```

## 实验组

Phase 2 现在固定比较三种方式：

```text
Direct
Image + Question -> VLM -> Answer
```

```text
Vanilla CoT
Image + Question -> VLM -> Think step by step -> Final Answer
```

```text
Structured CoT
Image + Question -> VLM -> Observation -> Reasoning -> Final Answer
```

注意：旧的 `exp02_cot*` 实际是 Structured CoT，因为 prompt 强制输出
`Observation / Reasoning / Final Answer`。Vanilla CoT 是本次新增的独立实验组。

## Prompt 区别

Vanilla CoT：

```text
Think step by step, then provide the final short answer.
Final Answer: ...
```

Structured CoT：

```text
Observation: describe only visual findings visible in the image.
Reasoning: explain how visual findings relate to the question.
Final Answer: provide only the short final answer.
```

两者都保留完整 `reasoning_output`，但指标只评估 `Final Answer:` 后的短答案。

## 4090D 运行顺序

先跑 20 条：

```bash
python scripts/run_direct_vlm.py --config configs/experiments/exp01_direct_vlm_qwen_7b_cuda_20.yaml
python scripts/run_cot.py --config configs/experiments/exp02_vanilla_cot_qwen_7b_4090d_20.yaml
python scripts/run_cot.py --config configs/experiments/exp02_structured_cot_qwen_7b_4090d_20.yaml
```

20 条稳定后跑 full：

```bash
python scripts/run_direct_vlm.py --config configs/experiments/exp01_direct_vlm_qwen_7b_cuda_full.yaml
python scripts/run_cot.py --config configs/experiments/exp02_vanilla_cot_qwen_7b_4090d_full.yaml
python scripts/run_cot.py --config configs/experiments/exp02_structured_cot_qwen_7b_4090d_full.yaml
```

## 对比分析

Direct vs Vanilla：

```bash
python scripts/compare_direct_vs_cot.py \
  --direct Results/exp01_direct_vlm_qwen_7b_cuda_full/predictions.jsonl \
  --cot Results/exp02_vanilla_cot_qwen_7b_4090d_full/predictions.jsonl \
  --output Results/compare_direct_vs_vanilla_cot_qwen_7b_4090d_full
```

Direct vs Structured：

```bash
python scripts/compare_direct_vs_cot.py \
  --direct Results/exp01_direct_vlm_qwen_7b_cuda_full/predictions.jsonl \
  --cot Results/exp02_structured_cot_qwen_7b_4090d_full/predictions.jsonl \
  --output Results/compare_direct_vs_structured_cot_qwen_7b_4090d_full
```

如果要比较 Vanilla vs Structured，可以把 `--direct` 临时理解成 baseline：

```bash
python scripts/compare_direct_vs_cot.py \
  --direct Results/exp02_vanilla_cot_qwen_7b_4090d_full/predictions.jsonl \
  --cot Results/exp02_structured_cot_qwen_7b_4090d_full/predictions.jsonl \
  --output Results/compare_vanilla_vs_structured_cot_qwen_7b_4090d_full
```

## 主要指标

```text
Accuracy
Token F1
BLEU-1
Latency
Input Tokens
Output Tokens
cot_helped
cot_hurt
both_correct
both_wrong
```

## 结果解释

如果 Vanilla CoT 高于 Direct，说明普通显式推理本身有帮助。

如果 Structured CoT 高于 Vanilla CoT，说明对医学 VQA 来说，把视觉观察和推理过程分开更可靠。

如果 Structured CoT 低于 Vanilla CoT，优先检查是否出现了错误 observation 或过度推理。Structured prompt 会让模型写更多内容，也可能放大错误链条。

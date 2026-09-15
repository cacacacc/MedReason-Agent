# Phase 1 / Phase 2 数值优化记录

本轮只优化 Phase 1 Direct VLM、Phase 2 CoT，以及 Phase 6 Adaptive Routing。

## 优化目标

Phase 1/2 的主要问题通常不是模型完全不会答，而是：

```text
1. yes/no 题输出了完整句子，exact match 被误判。
2. CoT 输出太长，Final Answer 被解释污染。
3. open question 输出句子，而不是 benchmark-style 短语。
```

因此本轮优化重点是短答案和归一化，而不是增加更多 agent。

## 已实现优化

### 1. yes/no 后处理增强

新增归一化：

```text
there is no / there are no / no evidence of / absent / negative -> no
present / visible / seen / evidence of -> yes
```

这类输出在医学 VQA 中很常见。它们语义上是 yes/no，但旧 exact match 可能无法稳定识别。

### 2. Direct Prompt 收紧

Direct VLM 现在更强调：

```text
Return only the short final answer.
For yes/no questions, answer exactly yes or no.
Do not add explanations.
Do not mention uncertainty unless the image is impossible to interpret.
```

### 3. CoT Prompt 收紧

Vanilla CoT 和 Structured CoT 都限制为短推理：

```text
Use at most two short reasoning steps.
Final Answer must still be a short benchmark-style answer.
Do not introduce diagnoses unless the question explicitly asks for diagnosis.
```

## 新增配置

Phase 1：

```bash
python scripts/run_direct_vlm.py --config configs/experiments/exp01_direct_vlm_qwen_7b_cuda_v2_100.yaml
python scripts/run_direct_vlm.py --config configs/experiments/exp01_direct_vlm_qwen_7b_cuda_v2_full.yaml
```

Phase 2 Vanilla CoT：

```bash
python scripts/run_cot.py --config configs/experiments/exp02_vanilla_cot_qwen_7b_4090d_v2_100.yaml
python scripts/run_cot.py --config configs/experiments/exp02_vanilla_cot_qwen_7b_4090d_v2_full.yaml
```

Phase 2 Structured CoT：

```bash
python scripts/run_cot.py --config configs/experiments/exp02_structured_cot_qwen_7b_4090d_v2_100.yaml
python scripts/run_cot.py --config configs/experiments/exp02_structured_cot_qwen_7b_4090d_v2_full.yaml
```

## 推荐运行顺序

```text
1. Direct v2 100
2. Structured CoT v2 100
3. Phase 6 v6 100
4. 如果 100 条结果不低于旧版本，再跑对应 full
5. Vanilla CoT v2 作为补充对照，不优先跑 full
```

## 判断标准

Direct v2：

```text
accuracy 应不低于旧 Direct full。
mean_token_f1 / mean_bleu_1 应略有提升。
```

Structured CoT v2：

```text
如果 accuracy 高于旧 Structured CoT，说明短 CoT 减少了推理漂移。
如果 accuracy 下降，说明模型需要更自由的 CoT，v2 不作为主结果。
```

Phase 6 v6：

```text
如果 v6_100 >= v5_100 的 57%，再跑 full。
如果 v6_full >= v5_full 的 55.88%，替换为新的 Phase 6 主结果。
```

# Phase 2 CoT Reasoning Results

## 实验设置

本阶段比较 `Direct Answer` 和 `Chain-of-Thought Prompt`，模型和数据保持一致，只改变 prompt 形式。

```text
模型：Qwen/Qwen2.5-VL-7B-Instruct
任务：VQA-RAD Medical VQA
split：test
样本数：451
Direct 结果：Results/exp01_direct_vlm_qwen_7b_cuda_full
CoT 结果：Results/exp02_cot_qwen_7b_cuda_full
对比输出：Results/compare_direct_vs_cot_qwen_7b_cuda_full
```

CoT prompt 要求模型输出：

```text
Observation
Reasoning
Final Answer
```

评估时只抽取 `Final Answer:` 后面的短答案计算指标，完整输出保存在 `reasoning_output`。

## 主结果

| Method | Samples | Accuracy | Correct | Token F1 | BLEU-1 |
|---|---:|---:|---:|---:|---:|
| Direct Answer | 451 | 49.22% | 222 / 451 | 55.41% | 53.97% |
| CoT Prompt | 451 | 43.02% | 194 / 451 | 51.19% | 49.24% |

结论：在当前 VQA-RAD full test 上，CoT 没有提升 baseline，反而使 accuracy 下降了 **6.21 个百分点**。

```text
Direct Accuracy = 0.4922
CoT Accuracy    = 0.4302
Delta           = -0.0621
```

## 运行成本

| Method | Mean Latency | Median Latency |
|---|---:|---:|
| Direct Answer | 4.36 s / sample | 2.57 s / sample |
| CoT Prompt | 72.41 s / sample | 69.90 s / sample |

CoT 的平均单题耗时约为 Direct 的 **16.6 倍**，中位耗时约为 Direct 的 **27.2 倍**。  
这说明在 RTX 4070 上，CoT 的额外 reasoning token 成本非常高。

## Direct vs CoT 逐题对比

| Outcome | Count | Meaning |
|---|---:|---|
| both_correct | 159 | Direct 和 CoT 都答对 |
| both_wrong | 194 | Direct 和 CoT 都答错 |
| cot_helped | 35 | Direct 错，CoT 对 |
| cot_hurt | 63 | Direct 对，CoT 错 |

CoT 改对了 35 题，但改错了 63 题，净损失是：

```text
35 - 63 = -28
```

这和总正确数从 222 降到 194 完全一致。

## 按答案类型分析

| Answer Type | Direct Accuracy | CoT Accuracy | Delta |
|---|---:|---:|---:|
| CLOSED | 70.59% | 64.71% | -5.88 pp |
| OPEN | 16.76% | 10.06% | -6.70 pp |

观察：

- CLOSED yes/no 问题上，CoT 仍然明显低于 Direct。
- OPEN 问题本来就难，CoT 进一步下降。
- OPEN 的 exact-match 会低估一些语义等价答案，例如 `Cerebrospinal Fluid (CSF)` 和 `CSF`。因此后续 GPT-based evaluation 很重要。

## 按问题类型分析

| Question Type | N | Direct Accuracy | CoT Accuracy | Delta |
|---|---:|---:|---:|---:|
| PRES | 163 | 55.83% | 49.69% | -6.14 pp |
| POS | 58 | 17.24% | 17.24% | 0.00 pp |
| ABN | 56 | 50.00% | 33.93% | -16.07 pp |
| SIZE | 45 | 57.78% | 64.44% | +6.67 pp |
| MODALITY | 33 | 54.55% | 45.45% | -9.09 pp |
| OTHER | 26 | 30.77% | 19.23% | -11.54 pp |
| PLANE | 26 | 80.77% | 50.00% | -30.77 pp |
| ATTRIB | 17 | 58.82% | 58.82% | 0.00 pp |
| ORGAN | 10 | 40.00% | 40.00% | 0.00 pp |
| COUNT | 6 | 33.33% | 50.00% | +16.67 pp |
| COLOR | 3 | 66.67% | 100.00% | +33.33 pp |

注意：`COUNT` 和 `COLOR` 样本数很小，不能过度解释。更稳定的趋势是：

- `SIZE` 问题上 CoT 有一定帮助。
- `ABN`、`PLANE`、`PRES` 问题上 CoT 更容易伤害结果。

## CoT 有帮助的例子

| sample_id | Question | Ground Truth | Direct | CoT |
|---|---|---|---|---|
| 1103 | Is the size of the ventricle abnormal? | Yes | No | Yes |
| 1112 | Is there a hypodense mass in the liver | Yes | No | Yes |
| 12 | Is there airspace consolidation on the left side? | Yes | No | Yes |

这些例子说明 CoT 有时能让模型重新检查视觉证据，尤其是存在性和大小相关问题。

## CoT 产生错误的例子

| sample_id | Question | Ground Truth | Direct | CoT |
|---|---|---|---|---|
| 1072 | In which lobe do you see an abnormal mass in the above images? | Right upper lobe | Right upper lobe | Right lower lobe |
| 1131 | Is the trachea midline? | Yes | Yes | No, the trachea is not midline. |
| 1144 | What is located immediately inferior to the right hemidiaphragm? | The liver | The liver. | Right kidney |
| 1157 | Does the liver show an enhancing mass or lesion? | No | No | Yes |

这些例子说明 CoT 可能出现“推理漂移”：模型在生成中间 reasoning 时引入额外假设，最后把原本正确的直接答案改错。

## 阶段结论

当前 Phase 2 的结论是：

```text
Naive CoT Prompt does not improve Qwen2.5-VL-7B on VQA-RAD full test.
```

更具体地说：

1. CoT accuracy 低于 Direct Answer。
2. CoT latency 明显更高，不适合直接作为默认 baseline。
3. CoT 对部分 `SIZE` 问题可能有帮助。
4. CoT 对 `ABN`、`PLANE`、`PRES` 等问题可能引入错误推理。
5. 后续应重点做 GPT-based evaluation 和错误类型标注，因为 exact-match 对 OPEN 答案偏严格。

## 下一步

Phase 3 的 Medical RAG 可以作为更合理的改进方向：不是让模型凭空多写 reasoning，而是在 reasoning 前提供外部医学证据。  
后续比较重点应从：

```text
Direct vs CoT
```

扩展为：

```text
Direct vs CoT vs RAG
```

同时需要单独分析：

```text
CoT hallucination
RAG evidence relevance
RAG 是否减少无依据推理
```


# Phase 2 CoT Reasoning Results

## 实验设置

本阶段比较 `Direct Answer`、`Vanilla CoT` 和 `Structured CoT`，模型和数据保持一致，只改变 prompt 形式。

```text
模型：Qwen/Qwen2.5-VL-7B-Instruct
任务：VQA-RAD Medical VQA
split：test
样本数：451
Direct 结果：Results/exp01_direct_vlm_qwen_7b_cuda_full
Vanilla CoT 结果：Results/exp02_vanilla_cot_qwen_7b_4090d_full
Structured CoT 结果：Results/exp02_structured_cot_qwen_7b_4090d_full
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
| Direct Answer | 451 | 49.89% | 225 / 451 | 56.09% | 54.59% |
| Vanilla CoT | 451 | 43.68% | 197 / 451 | 51.00% | 49.10% |
| Structured CoT | 451 | 43.90% | 198 / 451 | 51.24% | 49.52% |

结论：在本次 VQA-RAD full test 重跑中，两种 CoT 都没有提升 Direct baseline。
Structured CoT 比 Vanilla CoT 高 `0.22` 个百分点，但只多答对 1 条。

```text
Direct Accuracy           = 0.4989
Vanilla CoT Accuracy     = 0.4368
Structured CoT Accuracy  = 0.4390
Vanilla CoT delta         = -0.0621
Structured CoT delta      = -0.0599
Structured vs Vanilla     = +0.0022
```

## 运行成本

本次三个 full 实验的 `metrics.json` 没有写入 latency 字段，因此本次重跑只汇总答案质量指标，
不对运行成本做新的数值结论。下面的 latency 数字属于此前运行记录，不应与本次主表混用。

| Method | Mean Latency | Median Latency | 数据口径 |
|---|---:|---:|---|
| Direct Answer | 4.36 s / sample | 2.57 s / sample | 此前运行 |
| CoT Prompt | 72.41 s / sample | 69.90 s / sample | 此前运行 |

CoT 的平均单题耗时约为 Direct 的 **16.6 倍**，中位耗时约为 Direct 的 **27.2 倍**。  
这说明在 RTX 4070 上，CoT 的额外 reasoning token 成本非常高。

## Direct vs CoT 逐题对比

本次重跑没有生成 `compare_direct_vs_cot` 逐题对比文件，因此以下逐题统计属于此前运行，
仅作为历史参考；本次可确认的汇总结果以“主结果”表为准。

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

1. 本次重跑中 Vanilla CoT 和 Structured CoT accuracy 均低于 Direct Answer。
2. Structured CoT 只比 Vanilla CoT 高 `0.22` 个百分点，提升幅度很小。
3. 本次 `metrics.json` 未记录 latency，运行成本结论应以后续补充的 latency 统计为准。
4. 旧版逐题和分类型统计来自此前运行，不能直接当作本次重跑的分项结果。
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


# Phase 5 Verification Results

## 实验设置

本阶段在相同的 VQA-RAD test split 和 PMC 10k 检索语料上，比较三种验证策略：

```text
模型：Qwen/Qwen2.5-VL-7B-Instruct
本地模型路径：/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
Backend：qwen2_5_vl
数据集：VQA-RAD OSF public release
Split：test
样本数：451
训练 epochs：0
Backbone：frozen
Retriever：FAISS
Embedding：/root/autodl-tmp/models/bge-small-en-v1.5
Evidence top-k：5
```

三种 Phase 5 路径：

```text
Supervisor No Critic：候选答案，不做 verifier / critic / revision
Self-Reflection：同一个 reasoning agent 生成候选答案并自我反思
Separate Verifier：独立 verifier 检查 process-level reasoning，并决定是否 revise
```

## 主结果

| Method | Samples | Accuracy | Correct | Token F1 | BLEU-1 | Mean Latency | Agent Calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Supervisor No Critic | 451 | 21.73% | 98 / 451 | 30.29% | 27.94% | 9.82 s | 7.0 |
| Self-Reflection | 451 | 17.29% | 78 / 451 | 25.09% | 22.91% | 17.81 s | 8.82 |
| Separate Verifier | 451 | 20.18% | 91 / 451 | 28.22% | 26.03% | 21.09 s | 9.0 |

以 Supervisor No Critic 为 Phase 5 内部基线：

```text
Self-Reflection accuracy delta：-4.44 percentage points
Separate Verifier accuracy delta：-1.55 percentage points
```

三个方法都低于 Phase 4 Supervisor Multi-Agent full 的 `28.16%`，也低于 Phase 3
Knowledge RAG full 的 `22.84%`（Separate Verifier 和 Self-Reflection 低于它，No Critic
略低于它）。它们均未超过 Direct VLM 的 `49.22%`。

## Supervisor No Critic Full

```text
Experiment：exp05_supervisor_no_critic_qwen_7b_4090d_pmc10k_full
Accuracy：0.21729490022172948
Correct：98 / 451
Mean token F1：0.30293150612701936
Mean BLEU-1：0.2793911379895304
Mean latency：9818.13 ms
Mean total tokens：6389.65
Mean agent calls：7.0
```

该路径不调用 verifier，也不进行 revision：

```text
Verifier calls：0
Selective verification rate：0.0
Revision rate：0.0
Error corrections：0
Error regressions：0
Correct-answer preservation：1.0
Answer gate：DISABLED
```

候选答案和最终答案完全相同：

```text
Candidate accuracy：0.2173
Final accuracy：0.2173
Net correction：0
```

## Self-Reflection Full

```text
Experiment：exp05_self_reflection_qwen_7b_4090d_pmc10k_full
Accuracy：0.1729490022172949
Correct：78 / 451
Mean token F1：0.25093755337911744
Mean BLEU-1：0.22907103992558606
Mean latency：17805.94 ms
Mean total tokens：11013.18
Mean agent calls：8.82
```

候选答案与最终答案的变化：

```text
Candidate accuracy：0.21729490022172948
Final accuracy：0.1729490022172949
Initial errors：353
Initial correct：98
Error corrections：7
Error correction rate：0.01983
Error regressions：27
Error regression rate：0.27551
Correct-answer preservation：0.72449
Net correction：-20
```

Self-Reflection 对错误答案的修正只有 7 条，却把 27 条原本正确的候选答案改错，
因此最终 accuracy 比候选答案下降。它的 revision rate 为 `0.8248`，说明模型频繁修改
候选答案，但修改并没有带来可靠收益。

## Separate Verifier Full

```text
Experiment：exp05_separate_verifier_qwen_7b_4090d_pmc10k_full
Accuracy：0.2017738359201774
Correct：91 / 451
Mean token F1：0.28221719026250336
Mean BLEU-1：0.2602774043177489
Mean latency：21093.30 ms
Mean total tokens：11799.51
Mean agent calls：9.0
```

验证和 revision 情况：

```text
Verifier calls：451
Selective verification rate：1.0
Revision rate：1.0
Process verdict PASS：447
Process verdict REVISE：4
Error corrections：4
Error correction rate：0.01133
Error regressions：11
Error regression rate：0.11224
Correct-answer preservation：0.88776
Net correction：-7
```

候选答案和最终答案：

```text
Candidate accuracy：0.21729490022172948
Final accuracy：0.2017738359201774
Candidate -> final delta：-1.55 percentage points
Candidate hallucination rate：0.01774
Final hallucination rate：0.01552
Hallucination reduction：0.125
```

Separate Verifier 轻微降低了记录级 hallucination rate，但最终答案准确率也下降，
说明当前 verifier 更像是过程审计模块，尚未成为有效的答案纠错模块。

## Claim 与可靠性指标

三个 full 结果都记录了：

```text
Claim status records：859
HYPOTHESIS：859
OBSERVED / SUPPORTED / UNSUPPORTED / CONTRADICTED：0
Mean reasoning score：1.0
Mean tool selection accuracy：0.0
Mean persistent memory hits：0.0
Answer gate：DISABLED
```

这里的 `HYPOTHESIS` 和 `hallucination_rate` 反映的是当前 Phase 5 记录协议及规则指标，
不能直接等同于人工确认的医学幻觉。尤其是三个实验的 `tool_selection_accuracy=0.0`，
表明 Supervisor 的工具选择格式或 expected tools 对齐仍存在系统性问题，相关指标需要
先修正协议后再解释。

另外，Self-Reflection 和 Separate Verifier 的 process-level 字段显示：

```text
Self-Reflection：LOGICAL_ERROR detected 10，corrected 0
Separate Verifier：LOGICAL_ERROR detected 1，UNSUPPORTED_CLAIM detected 1，corrected 0
```

process-level 指标与最终答案级指标并不完全一致，论文中应分开报告。

## 与前序阶段对比

| Method | Accuracy | Mean Latency |
|---|---:|---:|
| Direct VLM | 49.22% | 4.36 s |
| CoT | 43.02% | 72.41 s |
| Phase 3 Knowledge RAG | 22.84% | 5.72 s |
| Phase 4 Fixed Multi-Agent | 39.02% | 11.61 s |
| Phase 4 Supervisor + PMC 10k | 28.16% | 14.86 s |
| Phase 5 Supervisor No Critic | 21.73% | 9.82 s |
| Phase 5 Self-Reflection | 17.29% | 17.81 s |
| Phase 5 Separate Verifier | 20.18% | 21.09 s |

当前结果没有显示 Phase 5 验证策略带来 accuracy 提升。Self-Reflection 和 Separate
Verifier 都比不验证的 No Critic 更慢，且最终准确率更低。

## 阶段结论

1. Supervisor No Critic 是三个 Phase 5 full 方法中 accuracy 最高的，达到 `21.73%`。
2. Self-Reflection 的修改行为不稳定，错误修正 `7` 条，但错误回归 `27` 条，净变化为 `-20`。
3. Separate Verifier 的过程验证调用全部执行，但只修正 `4` 条错误，同时产生 `11` 条回归。
4. Separate Verifier 将记录级 hallucination rate 从 `0.01774` 降到 `0.01552`，但没有提升最终答案 accuracy。
5. 三种 Phase 5 方法都未超过 Phase 4 Supervisor full 的 `28.16%` 或 Direct VLM 的 `49.22%`。
6. 当前 tool selection accuracy 为 `0.0`，claim status 几乎全部为 `HYPOTHESIS`，说明指标协议和
   Supervisor 输出格式仍需修正。

因此，Phase 5 当前证明了 verification pipeline 可以记录过程指标和答案转移，但还没有
证明 self-reflection 或 separate verifier 能提升 VQA-RAD 最终答案质量。

## 结果文件

```text
Results/exp05_supervisor_no_critic_qwen_7b_4090d_pmc10k_full/predictions.jsonl
Results/exp05_supervisor_no_critic_qwen_7b_4090d_pmc10k_full/metrics.json
Results/exp05_self_reflection_qwen_7b_4090d_pmc10k_full/predictions.jsonl
Results/exp05_self_reflection_qwen_7b_4090d_pmc10k_full/metrics.json
Results/exp05_separate_verifier_qwen_7b_4090d_pmc10k_full/predictions.jsonl
Results/exp05_separate_verifier_qwen_7b_4090d_pmc10k_full/metrics.json
```

`Results/` 被 `.gitignore` 忽略，因此本文件是 Phase 5 full 结果的 tracked summary。

## 100-Sample Tuning / Ablation Results

以下三组实验使用 VQA-RAD test split 的前 100 条样本，属于 tuning / ablation 结果，
不与 451 条 full test 结果混合。三组预测均为 100/100 条，唯一 sample_id 为 100，
空预测为 0。

| Method | Accuracy | Correct | Token F1 | BLEU-1 | Mean Latency | Candidate Accuracy | Final Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Supervisor No Critic | 36% | 36 / 100 | 38.15% | 37.80% | 8.19 s | 36% | 36% |
| Self-Reflection | 26% | 26 / 100 | 28.15% | 27.80% | 15.49 s | 36% | 26% |
| Separate Verifier | 34% | 34 / 100 | 36.15% | 35.80% | 17.44 s | 36% | 34% |

### Candidate 到 Final 的变化

三个方法的候选答案 accuracy 都是 `36%`，但后处理后的最终结果不同：

```text
Supervisor No Critic：不修改候选答案，36% -> 36%
Self-Reflection：修正 6 条，回归 16 条，36% -> 26%，net correction = -10
Separate Verifier：修正 4 条，回归 6 条，36% -> 34%，net correction = -2
```

Self-Reflection 的 revision rate 为 `0.88`，但正确答案保留率只有 `0.5556`；Separate
Verifier 的 revision rate 为 `1.0`，正确答案保留率为 `0.8333`。这表明在 100 条 tuning
样本上，Separate Verifier 比 Self-Reflection 稳定，但仍然没有超过不做 revision 的 No Critic。

### 可靠性与成本

```text
Supervisor No Critic：hallucination rate 0.00，verifier calls 0，agent calls 7.0
Self-Reflection：hallucination rate 0.88，verifier calls 0，agent calls 8.88
Separate Verifier：hallucination rate 1.00，verifier calls 100，agent calls 9.0
```

三组实验的 evidence 指标相同：

```text
Mean evidence quality：0.6106
Mean question coverage：0.6596
Mean answer coverage：0.2324
Evidence empty rate：0.0
```

Separate Verifier 每条样本都调用 verifier，平均耗时最高，为 `17.44 s`；Self-Reflection
平均耗时 `15.49 s`；No Critic 最快，为 `8.19 s`。

### 100-Sample 结论

```text
最高 accuracy：Supervisor No Critic，36%
最低 latency：Supervisor No Critic，8.19 s / sample
最少 answer regression：Supervisor No Critic，0 条
Self-Reflection net correction：-10
Separate Verifier net correction：-2
```

这组三组 100 条结果支持当前 Phase 5 的初步判断：验证和自反思增加了计算成本，
但在当前 prompt 和答案抽取协议下没有带来最终答案 accuracy 提升。Separate Verifier
比 Self-Reflection 更稳定，但仍低于 No Critic。需要在 451 条 full test 上进一步验证，
不能仅凭 100 条样本得出最终结论。

## Selective Verifier + Heuristic Routing

该实验使用 100 条 tuning 样本，将 selective verifier 与 heuristic routing 组合：

```text
Experiment：exp05_selective_verifier_heuristic_routing_qwen_7b_4090d_pmc10k_100
Samples：100
Accuracy：0.36
Correct：36 / 100
Mean token F1：0.3771
Mean BLEU-1：0.3735
Mean latency：12.19 s
Mean agent calls：5.17
```

候选答案到最终答案的变化：

```text
Candidate accuracy：0.39
Final accuracy：0.36
Verifier calls：71 / 100
Selective verification rate：0.71
Revision rate：0.70
Error corrections：4
Error correction rate：0.0164
Error regressions：6
Error regression rate：0.1026
Correct-answer preservation：0.8974
Net correction：-3
```

该实验的 evidence empty rate 为 `0.95`，mean evidence quality 仅 `0.0320`，
hallucination rate 为 `0.70`。虽然 heuristic routing 的 tool selection accuracy 为 `1.0`，
但大量样本跳过了检索，导致 selective verifier 缺少可用证据，最终 accuracy 低于候选答案。

因此，Selective Verifier + Heuristic Routing 在当前 100 条 tuning 结果中没有显示出
答案纠错收益；它适合用于分析“选择性验证在证据缺失时的行为”，不应作为当前最佳配置。

新增结果文件：

```text
Results/exp05_supervisor_no_critic_qwen_7b_4090d_pmc10k_100/predictions.jsonl
Results/exp05_supervisor_no_critic_qwen_7b_4090d_pmc10k_100/metrics.json
Results/exp05_self_reflection_qwen_7b_4090d_pmc10k_100/predictions.jsonl
Results/exp05_self_reflection_qwen_7b_4090d_pmc10k_100/metrics.json
Results/exp05_separate_verifier_qwen_7b_4090d_pmc10k_100/predictions.jsonl
Results/exp05_separate_verifier_qwen_7b_4090d_pmc10k_100/metrics.json
Results/exp05_selective_verifier_heuristic_routing_qwen_7b_4090d_pmc10k_100/predictions.jsonl
Results/exp05_selective_verifier_heuristic_routing_qwen_7b_4090d_pmc10k_100/metrics.json
```
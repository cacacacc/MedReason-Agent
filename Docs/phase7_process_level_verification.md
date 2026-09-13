# Phase 7：Process-Level Verification

Phase 7 的目标不是再提高 routing accuracy，而是研究：

```text
Verifier 是否能真正检查 reasoning process，而不是只对 final answer 做模糊评论。
```

本阶段继续使用 `SupervisorMultiAgent` 作为 candidate 生成器。Verifier 只发生在 candidate
answer 之后，不重新调用 Vision / Retrieval / Supervisor。

## 研究问题

核心问题：

```text
Separate Verifier 是否比 Self-Reflection 更可靠？
```

具体拆成三点：

```text
1. Verifier 能不能定位 candidate reasoning 中的错误 step？
2. Verifier 检出的错误是否对应真实错误答案？
3. Verifier 触发 revision 后，是 correction 多，还是 regression 多？
```

## 三组对照

Experiment 7A：

```text
No Verifier
vs
Self-Reflection
vs
Separate Verifier
```

三组都使用：

```text
backbone: Qwen2.5-VL-7B
candidate: SupervisorMultiAgent
training_epochs: 0
dataset: VQA-RAD test
scale: 300 samples
retriever: FAISS + BGE-small-en-v1.5 + PMC 10k
```

## Verifier 检查项

Separate Verifier 必须检查六类 process-level 信号：

```text
Observation Grounding
Evidence Support
Logical Consistency
Unsupported Claim
Contradiction
Overconfidence
```

输出必须包含：

```json
{
  "verdict": "PASS or REVISE",
  "step_results": [
    {
      "step_id": "obs_1 or evidence_1 or reason_1 or conclusion_1 or confidence_1",
      "status": "SUPPORTED or UNSUPPORTED or CONTRADICTED or UNCERTAIN",
      "error_type": null,
      "confidence": 0.0
    }
  ],
  "grounding_score": 0.0,
  "logical_consistency": 0.0,
  "confidence_alignment": 0.0,
  "error_types": [],
  "recommended_action": "PASS or REVISE",
  "revision_instruction": ""
}
```

## 指标

答案层指标：

```text
Accuracy
Candidate Accuracy
Final Accuracy
Correction Rate
Regression Rate
Correct Answer Preservation Rate
Hallucination Rate
```

过程验证指标：

```text
Verifier Precision
Verifier Recall
Verifier F1
Process error type counts
Process error recovery
Mean grounding score
Mean logical consistency
Mean confidence alignment
```

当前实现的 Verifier Precision / Recall / F1 使用弱监督定义：

```text
candidate_correct = false
-> 视为样本级 process error proxy

verifier 输出 REVISE / error_types / UNSUPPORTED / CONTRADICTED step
-> 视为 detected error
```

因此：

```text
True Positive:
candidate 错，verifier 检出错误

False Positive:
candidate 对，verifier 误报错误

False Negative:
candidate 错，verifier 没检出错误

True Negative:
candidate 对，verifier 没报错
```

这个定义不是人工 step-level gold label，但足够用于第一版 200-300 条调参。后续如果要更严谨，可以抽样人工标注 50 条 step-level error。

## 运行指令

No Verifier：

```bash
python scripts/run_phase7_process_verification.py --config configs/experiments/exp07_no_verifier_supervisor_multi_agent_qwen_7b_4090d_pmc10k_300.yaml
```

Self-Reflection：

```bash
python scripts/run_phase7_process_verification.py --config configs/experiments/exp07_self_reflection_supervisor_multi_agent_qwen_7b_4090d_pmc10k_300.yaml
```

Separate Verifier：

```bash
python scripts/run_phase7_process_verification.py --config configs/experiments/exp07_separate_verifier_supervisor_multi_agent_qwen_7b_4090d_pmc10k_300.yaml
```

当前 Separate Verifier 配置已升级为 v2：

```text
experiment_id: exp07_separate_verifier_supervisor_multi_agent_qwen_7b_4090d_pmc10k_300_v2
prompt_version: phase7_process_verification_v2
prompt_contract: separate_verifier_process_level_supervisor_candidate_parser_fixed_v2
result_dir: Results/exp07_separate_verifier_supervisor_multi_agent_qwen_7b_4090d_pmc10k_300_v2
```

这样可以避开旧 `predictions.jsonl` 的 resume 缓存，确保修复后的 parser 和 verifier prompt 被真正执行。

## 通过标准

Separate Verifier 比 Self-Reflection 更可靠，需要满足：

```text
verifier_f1 > self_reflection_f1
verifier_recall 明显高于 self_reflection_recall
error_regression_rate 不高于 self-reflection
process_error_type_counts 不是全 0
step_results 中能定位到具体 obs/evidence/reason/conclusion/confidence step
```

如果 Separate Verifier 的 final accuracy 没有提升，但 verifier_f1、error localization 和 hallucination/unsupported claim 指标更好，仍然可以作为有效结论：

```text
Separate Verifier is better as a process-level auditor, but not necessarily as an answer revision module.
```

这和 Phase 5 的负结果不冲突。Phase 5 说明全量 verifier 不适合直接改答案；Phase 7 则检验 verifier 是否能可靠发现和定位 reasoning 错误。

## 当前 300 条结果（修复前）

实验规模：

```text
Dataset: VQA-RAD test
Samples: 300
Candidate: SupervisorMultiAgent
Retriever: FAISS + BGE-small-en-v1.5 + PMC 10k
```

主结果（注意：Separate Verifier 的 revision 解析存在修复前不一致，需重跑后作为最终结果）：

| Method | Candidate Acc | Final Acc | Correction | Regression | Verifier F1 | Calls |
|---|---:|---:|---:|---:|---:|---:|
| No Verifier | 39.00% | 39.00% | 0.00% | 0.00% | 0.00% | 7.0 |
| Self-Reflection | 39.00% | 33.33% | 7.10% | 25.64% | 69.48% | 8.81 |
| Separate Verifier | 39.00% | 37.33% | 3.28% | 9.40% | 75.78% | 9.0 |

详细结果：

| Metric | No Verifier | Self-Reflection | Separate Verifier |
|---|---:|---:|---:|
| Final accuracy | 39.00% | 33.33% | 37.33% |
| Initial errors | 183 | 183 | 183 |
| Initial correct | 117 | 117 | 117 |
| Wrong -> Correct | 0 | 13 | 6 |
| Correct -> Wrong | 0 | 30 | 11 |
| Wrong -> Wrong | 183 | 170 | 177 |
| Correct -> Correct | 117 | 87 | 106 |
| Net correction | 0 | -17 | -5 |
| Correct preservation | 100.00% | 74.36% | 90.60% |
| Selective verification rate | 0.00% | 100.00% | 100.00% |
| Revision rate | 0.00% | 81.00% | 100.00% |
| Mean total tokens | 6555.82 | 11171.17 | 12060.52 |
| Mean latency | 7.95 s | 14.47 s | 16.97 s |

Verifier detection（修复前统计口径，重跑后以新 metrics 为准）：

| Method | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Verifier | 0.00% | 0.00% | 0.00% | 0 | 0 | 183 | 117 |
| Self-Reflection | 60.91% | 80.87% | 69.48% | 148 | 95 | 35 | 22 |
| Separate Verifier | 61.00% | 100.00% | 75.78% | 183 | 117 | 0 | 0 |

当前临时结论：

```text
1. Separate Verifier 的修复前 detection F1 高于 Self-Reflection：75.78% vs 69.48%。
2. Separate Verifier 的修复前 recall 达到 100%，但该数值受到 detection 统计口径影响。
3. Separate Verifier 的 regression 明显低于 Self-Reflection：9.40% vs 25.64%。
4. 但两者 final accuracy 都低于 No Verifier，说明当前 verifier 适合作为 auditor，不适合默认 revision。
5. Self-Reflection 过度修改最严重：修正 13 条，但误改 30 条，net correction=-17。
6. Separate Verifier 更可靠：修正 6 条，误改 11 条，net correction=-5，但仍不能直接作为答案改写模块。
```

实现注意：

```text
当前 Separate Verifier 出现 process_verdict=PASS 300，但 revision_rate=100% 的不一致信号。
原因是早期 parser 在 keep_decision=PASS 的 verifier schema 下，仍可能让 raw decision 字段覆盖 verdict。
代码已修正：Separate Verifier 现在优先使用 verdict / recommended_action。
Separate Verifier 配置已升级到 v2 输出目录，因此建议用修复后的代码重跑一次。
```

下一步调参方向：

```text
1. 不再跑 full verification + unconditional revision。
2. 保留 Separate Verifier 作为 process-level auditor。
3. 新增 gated revision：只有 verifier verdict=REVISE 且 error_types 非空时才允许 revision。
4. 优先报告 verifier_f1、recall、regression rate，而不是只追 final accuracy。
```

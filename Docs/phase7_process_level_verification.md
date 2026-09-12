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

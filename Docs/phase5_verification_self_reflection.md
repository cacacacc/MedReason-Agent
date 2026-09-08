# Phase 5：Verification + Self-Reflection

本阶段研究：

```text
candidate answer 生成之后，process-level verification 是否能修正错误、减少 hallucination、
unsupported claims 和错误 reasoning，并控制额外计算成本。
```

Phase 5 不训练模型：

```text
training_epochs: 0
backbone: Qwen2.5-VL-7B
变量: candidate 后是否加入 reflection / verifier / selective verifier
```

Phase 5 不做 Error-Conditioned Replanning。Verifier 可以检测并记录 error type，但不能重新调用 Vision / Retrieval / Supervisor。真正的 targeted replanning 放到后续 Phase。

## Phase 4 前提

所有 Phase 5 方法共享同一个 Phase 4 candidate 系统：

```text
Image + Question
-> Supervisor
-> Adaptive Route Execution
-> Optional One-Step Escalation
-> Candidate Reasoning Trace + Candidate Answer
```

Phase 5 从 `Candidate Reasoning Trace + Candidate Answer` 开始。

## 主实验路径

```text
No Check
-> final answer = candidate answer
```

```text
Self-Reflection
-> Candidate Reasoning + Answer
-> 同一个 Reasoning Agent 检查自己的 trace 和 answer
-> KEEP / REVISE
-> 如果 REVISE，同一个 Reasoning Agent 最多 revision 一次
-> Final Answer
```

```text
Separate Process-Level Verifier
-> Candidate Reasoning + Answer
-> 独立 Verifier Agent 做 process-level verification
-> PASS / REVISE
-> 如果 REVISE，Reasoning Agent 根据 verifier feedback 最多 revision 一次
-> Final Answer
```

Self-Reflection 和 Separate Verifier 是并行 baseline，不在主实验中串联。

## 严格边界

Self-Reflection 只允许原 Reasoning Agent 自检，不允许重新调用：

```text
Vision Agent
Retrieval Agent
Supervisor Agent
```

Separate Verifier 也不直接生成新答案。Verifier 只负责：

```text
detect
localize
classify
recommend correction
```

最终 revision 仍由 Reasoning Agent 完成。

## Self-Reflection

输入：

```text
Question
Vision Findings
Retrieved Evidence
Original Reasoning Steps
Candidate Answer
Candidate Confidence
```

检查：

```text
1. visual findings 是否被正确使用
2. evidence 是否支持 inference
3. 是否存在 logical leap
4. 是否存在 contradiction
5. 是否存在 unsupported claim
6. confidence 是否合理
```

输出 schema：

```json
{
  "decision": "KEEP | REVISE",
  "issues": [
    {
      "type": "PERCEPTION_ERROR | RETRIEVAL_ERROR | LOGICAL_ERROR | UNSUPPORTED_CLAIM | CONTRADICTION | OVERCONFIDENCE",
      "step_id": "",
      "description": ""
    }
  ],
  "reflection": "",
  "revision_instruction": ""
}
```

## Separate Verifier

Separate Verifier 使用：

```text
same backbone
different role prompt
independent context
```

Verifier 必须做五层 process-level verification：

```text
A. Observation Grounding
检查 reasoning 中的 observation 是否真的来自 Vision Findings。

B. Evidence Grounding
如果 route 调用了 Retrieval，检查 reasoning claim 是否被 retrieved evidence 支持。

C. Logical Consistency
检查 Observation -> Evidence -> Inference -> Conclusion 是否存在 logical leap、
invalid inference、causal overclaim 或 uncertainty amplification。

D. Contradiction
检查 reasoning trace 内部是否互相矛盾，以及是否和 Vision / Evidence 矛盾。

E. Confidence Alignment
检查 confidence 是否和视觉信心、证据强度、推理不确定性、多种可能假设相匹配。
```

Verifier 输出必须结构化：

```json
{
  "verdict": "PASS | REVISE",
  "step_results": [
    {
      "step_id": "obs_1",
      "status": "SUPPORTED | UNSUPPORTED | CONTRADICTED | UNCERTAIN",
      "error_type": null,
      "confidence": 0.91
    }
  ],
  "grounding_score": 0.82,
  "logical_consistency": 0.75,
  "confidence_alignment": 0.61,
  "error_types": [
    "LOGICAL_ERROR"
  ],
  "recommended_action": "REVISE",
  "revision_instruction": "Remove the unsupported causal inference."
}
```

`error_types` 覆盖：

```text
PERCEPTION_ERROR
RETRIEVAL_ERROR
LOGICAL_ERROR
UNSUPPORTED_CLAIM
CONTRADICTION
OVERCONFIDENCE
```

这些 error type 在 Phase 5 只用于分析和 textual revision，不用于重新调用上游 Agent。Phase 6 可以直接复用 `process_error_types` 做 Error-Conditioned Replanning。

## Experiment 5A

Research Question：

```text
Does independent process-level verification improve error correction compared with self-reflection?
```

比较：

```text
No Check
Self-Reflection
Separate Process-Level Verifier
```

控制变量：

```text
same Phase 4 Supervisor
same backbone
same dataset
same split
same candidate route policy
same token budget policy
same maximum one revision
same evaluator
```

## Experiment 5C

Research Question：

```text
Does step-level process verification detect and repair reasoning failures better than final-answer-only checking?
```

比较：

```text
Outcome-Level Verifier
只看 Question + Candidate Answer

Process-Level Verifier
看 Vision Findings + Retrieved Evidence + Reasoning Steps + Candidate Answer
```

## Experiment 5D

Research Question：

```text
Can selective verification reduce verifier calls and token cost while keeping reliability close to full verification?
```

比较：

```text
No Verification
100% Process Verification
Selective Process Verification
```

第一版 selective rule：

```text
verify if:
- candidate confidence < 0.6
- claim_verification_status is UNSUPPORTED / CONTRADICTED
- claim_statuses contain UNSUPPORTED / CONTRADICTED
- Retrieval route has no retrieved evidence
- reasoning contains explicit uncertainty such as uncertain / insufficient / limited / cannot
- deterministic answer gate used REVISED / FALLBACK
- state compression ratio > 0.9
```

这个 gate 只决定是否调用后置 Verifier，不触发上游重跑。

## 指标

主指标：

```text
Accuracy
Correction Rate
Regression Rate
Correct Answer Preservation Rate
Hallucination Rate
Unsupported Claim Rate
Tokens
Latency
Agent Calls
Selective Verification Rate
```

Correction Rate：

```text
Wrong -> Correct / Initially Wrong
```

Regression Rate：

```text
Correct -> Wrong / Initially Correct
```

Correct Answer Preservation Rate：

```text
Correct -> Correct / Initially Correct
```

Paired transition 必须保存：

```text
Wrong -> Correct
Correct -> Wrong
Wrong -> Wrong
Correct -> Correct
```

Error-type recovery：

```text
Detected
Corrected
Uncorrected
Regressed
Recovery Rate
```

## 输出字段

每条 prediction 会保存：

```text
candidate_prediction
candidate_correct
prediction
correct
process_feedback_output
process_feedback
process_verdict
process_recommended_action
process_step_results
process_error_types
process_grounding_score
process_logical_consistency
process_confidence_alignment
revision_output
revision_applied
selective_verification_applied
candidate_error_attribution
error_attribution
tool_calls
latency_ms
input_tokens
output_tokens
```

## AutoDL 运行

先跑 20 条：

```bash
python scripts/run_phase5_verification.py --config configs/experiments/exp05_supervisor_no_critic_qwen_7b_4090d_pmc10k_20.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_self_reflection_qwen_7b_4090d_pmc10k_20.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_outcome_verifier_qwen_7b_4090d_pmc10k_20.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_separate_verifier_qwen_7b_4090d_pmc10k_20.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_selective_verifier_qwen_7b_4090d_pmc10k_20.yaml
```

20 条稳定后跑 full：

```bash
python scripts/run_phase5_verification.py --config configs/experiments/exp05_supervisor_no_critic_qwen_7b_4090d_pmc10k_full.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_self_reflection_qwen_7b_4090d_pmc10k_full.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_outcome_verifier_qwen_7b_4090d_pmc10k_full.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_separate_verifier_qwen_7b_4090d_pmc10k_full.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_selective_verifier_qwen_7b_4090d_pmc10k_full.yaml
```

## 结果解释

如果 `Self-Reflection` correction rate 提升但 regression rate 也高，说明自检能发现一部分错误，但容易把原本正确答案改错。

如果 `Separate Verifier` 比 `Self-Reflection` 有更高 correction rate 和更低 regression rate，说明独立 verifier role 更适合发现 unsupported / inconsistent reasoning。

如果 `Process-Level Verifier` 优于 `Outcome-Level Verifier`，说明只看 final answer 不够，必须检查 Vision / Evidence / Reasoning trace。

如果 `Selective Verifier` 的 verification rate 明显低于 100%，但 accuracy / hallucination 接近 Full Verifier，说明 selective verification 有 reliability-efficiency trade-off 价值。

如果错误主要集中在 `PERCEPTION_ERROR` 或 `RETRIEVAL_ERROR`，Phase 5 textual revision 通常很难修复，这正好引出 Phase 6 的 Error-Conditioned Replanning。

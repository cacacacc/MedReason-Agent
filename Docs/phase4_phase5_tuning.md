# Phase 4 / Phase 5 调参协议

当前本地 `Results/` 里 Phase 4 和 Phase 5 都只有 mock 20 条结果，不能作为真实调参依据。
mock 结果只能说明 pipeline 和输出 schema 能跑通。4090D 上应使用 100 条样本做调参。

## Phase 4：调架构和路由

Phase 4 不调 verifier prompt。它只回答：

```text
Fixed Multi-Agent 是否优于 Structured CoT？
Supervisor 是否优于 Fixed Multi-Agent？
retrieval top_k 多少合适？
dynamic_routing / answer_gate 是否值得打开？
```

推荐 100 条调参顺序：

```bash
python scripts/run_multi_agent.py --config configs/experiments/exp04_fixed_multi_agent_qwen_7b_4090d_100.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_100.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_k3_100.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_k10_100.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_dynamic_only_qwen_7b_4090d_pmc10k_100.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_gate_only_qwen_7b_4090d_pmc10k_100.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_dynamic_gate_qwen_7b_4090d_pmc10k_100.yaml
```

选择规则：

```text
1. 先看 k=3 / k=5 / k=10：
   accuracy 接近时选 token 更少、latency 更低的配置。

2. dynamic_only 如果 accuracy 不升，且 tool_selection_accuracy / latency 变差：
   full 主实验不打开 dynamic_routing。

3. gate_only 如果 hallucination_rate 降低且 accuracy 不掉：
   可以把 answer_gate 放到 Phase5 或 Phase6 主线。

4. dynamic_gate 如果优于 dynamic_only 和 gate_only：
   说明 routing + gate 有协同；否则不要把两个变量混进主实验。
```

Phase 4 full 只跑最终选中的 1-2 个配置，不要把所有 100 条 ablation 都扩成 full。

## Phase 5：调验证策略

Phase 5 不重新调用 Vision / Retrieval / Supervisor。它只回答：

```text
Self-Reflection 是否有帮助？
Separate Process-Level Verifier 是否比 Self-Reflection 更好？
256 tokens 是否足够，还是需要 384 tokens？
```

推荐 100 条调参顺序：

```bash
python scripts/run_phase5_verification.py --config configs/experiments/exp05_supervisor_no_critic_qwen_7b_4090d_pmc10k_100.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_self_reflection_qwen_7b_4090d_pmc10k_100.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_separate_verifier_qwen_7b_4090d_pmc10k_100.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_self_reflection_qwen_7b_4090d_pmc10k_max384_100.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_separate_verifier_qwen_7b_4090d_pmc10k_max384_100.yaml
```

选择规则：

```text
1. No Check 是 candidate baseline。

2. Self-Reflection 如果 correction_rate 提升但 regression_rate 也高：
   说明自检会修错一些样本，但稳定性不足。

3. Separate Verifier 如果 correction_rate 更高、regression_rate 更低：
   Phase5 full 主实验优先选择 Separate Verifier。

4. 384 tokens 只有在 revision_output 明显被截断，或 correction_rate 明显高于 256 时才保留。
   否则主实验使用 256，节省成本。
```

Phase 5 full 最少跑：

```bash
python scripts/run_phase5_verification.py --config configs/experiments/exp05_supervisor_no_critic_qwen_7b_4090d_pmc10k_full.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_self_reflection_qwen_7b_4090d_pmc10k_full.yaml
python scripts/run_phase5_verification.py --config configs/experiments/exp05_separate_verifier_qwen_7b_4090d_pmc10k_full.yaml
```

如果 100 条显示 384 tokens 明显更好，再复制 full 配置改 `max_new_tokens: 384`。

## 主要看哪些指标

Phase 4：

```text
accuracy
mean_token_f1
mean_evidence_quality_score
hallucination_rate
mean_tool_selection_accuracy
answer_gate_counts
latency_ms
input_tokens / output_tokens
```

Phase 5：

```text
candidate_accuracy
final_accuracy
error_correction_rate
error_regression_rate
correct_answer_preservation_rate
net_correction
revision_rate
selective_verification_rate
process_error_type_counts
process_error_recovery
candidate_hallucination_rate
final_hallucination_rate
```

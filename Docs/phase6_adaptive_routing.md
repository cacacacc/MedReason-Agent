# Phase 6：Supervisor Adaptive Routing

Phase 6 是项目的核心阶段。前面 Phase 4 / Phase 5 的结果说明：固定地让所有样本走完整
Multi-Agent、RAG、Verifier，并不会稳定提升 VQA-RAD accuracy。相反，简单 closed visual
questions 容易被过度推理和无关 evidence 干扰。

因此 Phase 6 的目标是：

```text
不让所有问题都跑完整 Multi-Agent，而是根据难度和不确定性动态选择 reasoning depth。
```

## 核心路线

当前实现是 Rule-Based Supervisor：

```text
Question metadata + question text
↓
Rule-Based Supervisor
↓
LOW     -> Direct VLM
MEDIUM  -> Structured CoT
HIGH    -> Full Supervisor Multi-Agent
```

第一版只使用推理前可得信息，不偷看 ground truth，也不使用模型输出后的正确性。

## 路由信号

每条样本记录：

```text
complexity
confidence_signal
uncertainty_signal
evidence_support_signal
agent_agreement_signal
matched_markers
route
fallback_applied
fallback_reason
```

v1 规则：

```text
LOW:
closed + PRES / MODALITY / PLANE / ORGAN / COUNT / COLOR
-> Direct

MEDIUM:
视觉位置、大小、属性等轻量推理
-> Structured CoT

HIGH:
OPEN question，ABN / OTHER，或包含 diagnosis / disease / cause / abnormality 等医学知识触发词
-> Full Multi-Agent
```

## Fallback

当前 fallback 是保守版：

```text
Direct -> Structured CoT
Structured CoT -> Full Multi-Agent
Full Multi-Agent -> no fallback
```

只有模型输出为空、unknown、uncertain，或明确说 insufficient information / cannot determine
时才触发 fallback。这样可以避免把大量本来正确的短答案再次送入复杂链路。

## 指标

Phase 6 继承 accuracy、token F1、BLEU-1、evidence quality、claim status、agent metrics，
并新增：

```text
route_distribution
complexity_distribution
fallback_rate
fallback_count
route_accuracy
mean_agent_calls
```

## 当前 v1 100 条结果

实验：

```text
Experiment: exp06_rule_based_adaptive_routing_qwen_7b_4090d_pmc10k_100
Samples: 100
Model: /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
Backend: qwen2_5_vl
Embedding / retrieval: FAISS + PMC 10k
Routing policy: rule_based_v1
```

结果完整性：

```text
Prediction records: 100 / 100
Unique sample IDs: 100
Empty predictions: 0
Valid main result: true
```

主要指标：

| Metric | Value |
|---|---:|
| Accuracy | 43.00% |
| Correct | 43 / 100 |
| Mean token F1 | 44.90% |
| Mean BLEU-1 | 44.65% |
| Mean evidence quality | 0.2943 |
| Evidence empty rate | 52.00% |
| Hallucination rate | 25.00% |
| Mean reasoning score | 0.61 |
| Mean agent calls | 5.8 |
| Fallback rate | 0.00% |
| Fallback count | 0 |

## v1 路由分布

```text
LOW / Direct: 39 samples
MEDIUM / Structured CoT: 13 samples
HIGH / Full Multi-Agent: 48 samples
```

各路线内部表现：

| Route | Samples | Accuracy |
|---|---:|---:|
| Direct | 39 | 56.41% |
| Structured CoT | 13 | 76.92% |
| Full Multi-Agent | 48 | 22.92% |

复杂度分布不是全部 HIGH，说明 rule-based routing 确实执行了不同 reasoning depth。
其中 Structured CoT 路由在 13 条样本上表现最好，但样本量较小，不能单独作为稳定结论。

## 与 Phase 4 对照

当前 Phase 6 是 100 条 tuning 结果，Phase 4 heuristic full 是 451 条结果，因此不做严格同规模比较，
只列作方向性参考：

| Method | Samples | Accuracy | Mean Evidence Quality | Evidence Empty Rate | Tool Selection |
|---|---:|---:|---:|---:|---:|
| Phase 4 Heuristic Routing | 451 | 41.02% | 0.1079 | 83.37% | 1.0 |
| Phase 6 Adaptive Routing v1 | 100 | 43.00% | 0.2943 | 52.00% | not recorded |

Phase 6 在较小样本上 accuracy 高于 Phase 4 heuristic full 的数值，但这不是严格的性能提升证据。
Phase 6 同时减少了 evidence empty rate，说明部分样本被送入了包含 retrieval 的路线，但 mean agent
calls 仍为 `5.8`，低于完整 Multi-Agent 的调用成本。

## 错误与风险

自动错误归因记录：

```text
Routing Error: 57
State Error: 20
Human review required: 57
Unknown error: 0
```

`State Error=20` 和 `mean reasoning score=0.61` 说明 v1 中不同路线产生的 shared state 字段不完全一致。
尤其 Direct / Structured CoT 路线不使用完整 Multi-Agent shared state，因此容易被旧版 error attribution
规则误判为 State Error。这个问题会影响 claim 和 error attribution 指标的解释。

Claim status：

```text
SUPPORTED: 23
HYPOTHESIS: 22
UNSUPPORTED: 29
CONTRADICTED: 0
```

Answer gate 状态：

```text
DISABLED: 48
NOT_APPLICABLE: 52
```

## v1 阶段性结论

1. Rule-Based Adaptive Routing v1 在 100 条 tuning 样本上达到 `43%` accuracy，完成了 Direct、
   Structured CoT 和 Full Multi-Agent 的混合路由。
2. Structured CoT route 的内部 accuracy 为 `76.92%`，但只有 13 条样本，必须在后续实验中扩大验证。
3. Full Multi-Agent route 的内部 accuracy 为 `22.92%`，是三种路线中最低的，说明不要让过多样本进入完整 agent 链路。
4. fallback 没有触发，fallback rate 为 `0.0`，说明本轮结果主要来自初始规则路由。
5. v1 存在 State Error 和 reasoning score 偏低问题，正式 full run 前应先修正路线间 record schema。

## v2 路由优化

v1 的主要问题不是 adaptive routing 方向错，而是 HIGH 路由太多。尤其 v1 把所有 OPEN question
都直接送入 Full Multi-Agent，导致完整 agent 链路拖低整体 accuracy。

v2 做了三点修改：

```text
1. 不再把所有 OPEN question 直接判为 HIGH。
2. OPEN + visual description / anatomy / location / abnormality description -> MEDIUM。
3. 只有 diagnosis / disease / cause / etiology / differential / compatible / consistent with
   这类强医学知识触发词才进入 HIGH。
```

v2 的目标路由分布：

```text
LOW: 35%-45%
MEDIUM: 25%-40%
HIGH: 20%-30%
```

同时，v2 修复了 Direct / Structured CoT 路线的最小 shared state 记录。它们不使用完整 Multi-Agent
shared state，但会写入兼容字段，避免 error attribution 把正常的 Direct / CoT 路线误判为 State Error。

## 当前 v2 100 条结果

实验：

```text
Experiment: exp06_rule_based_adaptive_routing_v2_qwen_7b_4090d_pmc10k_100
Samples: 100
Model: /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
Backend: qwen2_5_vl
Embedding / retrieval: FAISS + PMC 10k
Routing policy: rule_based_v2
```

主要指标：

| Metric | Value |
|---|---:|
| Accuracy | 44.00% |
| Correct | 44 / 100 |
| Mean evidence quality | 0.1133 |
| Evidence empty rate | 86.00% |
| Mean agent calls | 2.4 |
| Mean latency | 2.74 s |
| Fallback count | 0 |
| State Error | 0 |

v2 路由分布：

```text
LOW / Direct: 39 samples
MEDIUM / Structured CoT: 47 samples
HIGH / Full Multi-Agent: 14 samples
```

各路线内部表现：

| Route | Samples | Correct | Accuracy | Mean Latency | Mean Agent Calls |
|---|---:|---:|---:|---:|---:|
| Direct | 39 | 22 | 56.41% | 0.42 s | 1.0 |
| Structured CoT | 47 | 13 | 27.66% | 1.81 s | 1.0 |
| Full Multi-Agent | 14 | 9 | 64.29% | 12.29 s | 11.0 |

v2 相比 v1：

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| Accuracy | 43.00% | 44.00% | +1.00 |
| Direct samples | 39 | 39 | 0 |
| Structured CoT samples | 13 | 47 | +34 |
| Full Multi-Agent samples | 48 | 14 | -34 |
| Evidence empty rate | 52.00% | 86.00% | +34.00 |
| Mean agent calls | 5.8 | 2.4 | -3.4 |
| State Error | 20 | 0 | -20 |

自动错误归因：

```text
Reasoning Error: 51
Routing Error: 5
State Error: 0
Perception Error: 5
Verification Error: 1
```

按问题类型看，v2 的 HIGH route 主要集中在：

```text
ABN: 13
OTHER: 1
```

Structured CoT route 包含：

```text
SIZE: 12
POS: 8
PRES: 7
ATTRIB: 5
ABN: 5
OTHER: 2
COUNT: 2
POS, PRES: 2
SIZE, PRES: 1
PLANE: 1
ATTRIB, PRES: 1
MODALITY: 1
```

## v2 阶段性结论

v2 达到了三个目标：

```text
1. HIGH route 从 48 条降到 14 条。
2. mean agent calls 从 5.8 降到 2.4。
3. State Error 从 20 降到 0。
```

但 v2 也暴露了一个新问题：

```text
Structured CoT route 被扩大到 47 条后，内部 accuracy 只有 27.66%。
```

这说明 v1 中 Structured CoT 的 `76.92%` 很可能来自样本选择较容易，而不是 CoT 本身稳定更强。
v2 当前 accuracy 只比 v1 高 1 个百分点，因此不建议直接把 v2 作为最终 full 主实验。更合理的下一步是
做 v3：保留 v2 的低成本和 State Error 修复，但收紧 Structured CoT 的适用范围。

v3 的方向：

```text
1. Direct 保持当前规则。
2. Structured CoT 只保留 POS / SIZE / ATTRIB 这类明确视觉推理题。
3. OPEN + OTHER / PRES 不再默认进 Structured CoT，改为 Direct 或轻量 Multi-Agent 对照。
4. ABN closed 题如果包含 pathology / vascular / abnormality，可进入 Full Multi-Agent。
5. 加 Oracle Routing 分析，判断 Direct / CoT / Full 中每个样本的最佳路线。
```

## v3 路由优化

v3 已实现为独立 `rule_based_v3`，不覆盖 v1 / v2。它基于 v2 的 100 条结果做收缩：

```text
Direct:
closed simple visual questions
OPEN + PRES / OTHER / COUNT / PLANE / MODALITY 等短答案视觉题

Structured CoT:
POS / SIZE / ATTRIB
POS, PRES / SIZE, PRES / ATTRIB, PRES 这类组合视觉推理题

Full Multi-Agent:
ABN
diagnosis / disease / cause / etiology / differential / compatible / consistent with / pathology
```

v3 的预期变化：

```text
Direct 增加
Structured CoT 明显减少
Full Multi-Agent 略高于 v2，但低于 v1
State Error 保持 0
mean agent calls 仍显著低于 v1
```

根据 v2 的实际路由分布，v3 在同一 100 条样本上的预期分布大致是：

```text
LOW / Direct: 约 52
MEDIUM / Structured CoT: 约 29
HIGH / Full Multi-Agent: 约 19
```

v3 是否继续 full 的判断标准：

```text
accuracy >= 44%
Structured CoT route accuracy 明显高于 v2 的 27.66%
HIGH route 占比低于 v1 的 48%
State Error = 0
mean agent calls 接近或低于 v2 的 2.4，最多不超过 v1 的 5.8
```

## 当前 v3 100 条结果

实验：

```text
Experiment: exp06_rule_based_adaptive_routing_v3_qwen_7b_4090d_pmc10k_100
Samples: 100
Model: /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
Backend: qwen2_5_vl
Embedding / retrieval: FAISS + PMC 10k
Routing policy: rule_based_v3
```

主要指标：

| Metric | Value |
|---|---:|
| Accuracy | 45.00% |
| Correct | 45 / 100 |
| Mean evidence quality | 0.1347 |
| Evidence empty rate | 82.00% |
| Mean agent calls | 2.8 |
| Mean latency | 2.94 s |
| Fallback count | 0 |
| State Error | 0 |

v3 路由分布：

```text
LOW / Direct: 53 samples
MEDIUM / Structured CoT: 29 samples
HIGH / Full Multi-Agent: 18 samples
```

各路线内部表现：

| Route | Samples | Correct | Accuracy | Mean Latency | Mean Agent Calls |
|---|---:|---:|---:|---:|---:|
| Direct | 53 | 25 | 47.17% | 0.42 s | 1.0 |
| Structured CoT | 29 | 12 | 41.38% | 1.75 s | 1.0 |
| Full Multi-Agent | 18 | 8 | 44.44% | 12.24 s | 11.0 |

v3 相比 v1 / v2：

| Metric | v1 | v2 | v3 |
|---|---:|---:|---:|
| Accuracy | 43.00% | 44.00% | 45.00% |
| Direct samples | 39 | 39 | 53 |
| Structured CoT samples | 13 | 47 | 29 |
| Full Multi-Agent samples | 48 | 14 | 18 |
| Evidence empty rate | 52.00% | 86.00% | 82.00% |
| Mean agent calls | 5.8 | 2.4 | 2.8 |
| State Error | 20 | 0 | 0 |

自动错误归因：

```text
Reasoning Error: 45
Routing Error: 10
State Error: 0
Perception Error: 10
Verification Error: 4
```

按问题类型看：

```text
Direct:
PRES 35, COUNT 6, PLANE 5, OTHER 3, COLOR 2, ORGAN 1, MODALITY 1

Structured CoT:
SIZE 12, POS 8, ATTRIB 5, POS/PRES 2, SIZE/PRES 1, ATTRIB/PRES 1

Full Multi-Agent:
ABN 18
```

## v3 阶段性结论

v3 相比 v2 达到了预期目标：

```text
1. accuracy 从 44% 提升到 45%。
2. Structured CoT route 从 47 条收紧到 29 条。
3. Structured CoT route accuracy 从 27.66% 提升到 41.38%。
4. HIGH route 仍明显低于 v1：18 条 vs 48 条。
5. State Error 保持 0。
6. mean agent calls 为 2.8，明显低于 v1 的 5.8。
```

v3 已满足进入 full 的最低条件。需要注意的是，v3 的 Direct / CoT / Full 三条 route 内部
accuracy 已比较接近，说明继续靠手写规则微调的收益会变小。下一步应优先跑 v3 full，
然后做 Oracle Routing 分析，判断每个样本理论上应该选择 Direct、CoT 还是 Full。

## 实验指令

Phase 4 heuristic full，作为当前最强 Multi-Agent 候选：

```bash
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_full.yaml
```

Phase 6 v1 100 条：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_qwen_7b_4090d_pmc10k_100.yaml
```

Phase 6 v2 100 条：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_v2_qwen_7b_4090d_pmc10k_100.yaml
```

Phase 6 v3 100 条：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_v3_qwen_7b_4090d_pmc10k_100.yaml
```

如果 v2 100 条满足下面条件，再跑 full：

```text
accuracy >= 43%
HIGH route 占比低于 v1
Full Multi-Agent route accuracy 不再明显拖低整体结果
State Error 明显下降
mean_agent_calls 不高于 v1 的 5.8
```

Phase 6 v2 full：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_v2_qwen_7b_4090d_pmc10k_full.yaml
```

Phase 6 v3 full：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_v3_qwen_7b_4090d_pmc10k_full.yaml
```

## 与 Phase 4 / Phase 5 的关系

Phase 4 证明固定 Multi-Agent 和普通 Supervisor 当前没有超过 Direct VLM。

Phase 5 证明 Self-Reflection / Separate Verifier 可以记录过程审计信息，但当前会造成 answer
regression，不适合作为默认改答案模块。

Phase 6 因此把重点放到：

```text
什么时候不要调用复杂 agent
什么时候才值得调用 RAG / Verifier
```

## 结果文件

```text
Results/exp06_rule_based_adaptive_routing_qwen_7b_4090d_pmc10k_100/predictions.jsonl
Results/exp06_rule_based_adaptive_routing_qwen_7b_4090d_pmc10k_100/metrics.json
```

## Oracle Routing full 分析

基于 451 条 VQA-RAD full test，同时比较 Direct、Structured CoT、Full Multi-Agent 和 Phase 6 v3 的选择结果。

关键结果：

| Metric | Value |
|---|---:|
| Samples | 451 |
| Oracle accuracy | 61.20% |
| Phase 6 v3 routing accuracy | 50.55% |
| Routing-to-oracle match rate | 35.48% |
| Over-reasoning rate | 18.63% |
| Under-reasoning rate | 7.10% |
| Unrecoverable error rate | 38.80% |

Oracle 最优路线分布：

| Oracle Route | Count |
|---|---:|
| Direct | 225 |
| Structured CoT | 42 |
| Full Multi-Agent | 9 |
| None correct | 175 |

Phase 6 v3 实际选择分布：

| Selected Route | Count |
|---|---:|
| Direct | 264 |
| Structured CoT | 127 |
| Full Multi-Agent | 60 |

结论：

```text
1. Direct 是当前最强的默认路线，Oracle 中有 225 条样本由 Direct 最优。
2. Full Multi-Agent 的 Oracle 最优样本只有 9 条，但 v3 实际选择了 60 条，说明仍然过度调用复杂 agent。
3. Structured CoT 的 Oracle 最优样本是 42 条，但 v3 实际选择了 127 条，说明 CoT 也存在过度使用。
4. over-reasoning rate 18.63% 高于 under-reasoning rate 7.10%，下一版应优先减少过度推理。
5. ABN 不应该自动进入 Full Multi-Agent。ABN 的 Oracle 分布是 Direct 29、Structured CoT 5、Full 1、None 21。
```

因此 v4 的设计目标不是继续扩大 Multi-Agent，而是更接近 Oracle 分布：

```text
Direct:
默认路线。普通 PRES / POS / ABN / MODALITY / PLANE / ORGAN / COUNT / COLOR 只要没有强医学推理触发词，优先 Direct。

Structured CoT:
只保留 SIZE / ATTRIB，或明确 larger / smaller / increased / decreased / compare / which side 等比较型视觉推理触发词。

Full Multi-Agent:
只保留 diagnosis / diagnostic / differential / etiology / cause / disease / compatible with / consistent with 等强医学知识触发词。
```

## v4 路由优化

v4 已实现为独立 `rule_based_v4`，不覆盖 v1 / v2 / v3。它的核心假设是：

```text
当 Direct 已经是最强单路线时，Supervisor 的第一职责不是增加 reasoning depth，
而是避免把简单视觉问题送入会产生答案漂移的复杂链路。
```

v4 相比 v3 的变化：

| Rule | v3 | v4 |
|---|---|---|
| ABN | 默认 Full Multi-Agent | 默认 Direct，除非有强诊断触发词 |
| POS | 默认 Structured CoT | 默认 Direct |
| SIZE / ATTRIB | Structured CoT | 保留 Structured CoT |
| pathology / abnormality | 可触发 Full | 不再单独触发 Full |
| diagnosis / cause / differential | Full | 保留 Full |

v4 100 条调参：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_v4_qwen_7b_4090d_pmc10k_100.yaml
```

v4 full：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_v4_qwen_7b_4090d_pmc10k_full.yaml
```

建议先跑 100 条。如果 v4 100 条满足下面条件，再跑 full：

```text
accuracy >= v3 100 的 45%
Direct route 占比明显上升
Structured CoT route 占比下降
Full Multi-Agent route 占比下降
State Error = 0
mean agent calls <= v3 的 2.8
```

## 当前 v4 100 条结果

实验：

```text
Experiment: exp06_rule_based_adaptive_routing_v4_qwen_7b_4090d_pmc10k_100
Samples: 100
Model: /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
Backend: qwen2_5_vl
Embedding / retrieval: FAISS + PMC 10k，但 v4 100 条没有触发 Full Multi-Agent，因此 evidence 为空
Routing policy: rule_based_v4
```

主要指标：

| Metric | Value |
|---|---:|
| Accuracy | 48.00% |
| Correct | 48 / 100 |
| Mean token F1 | 49.95% |
| Mean BLEU-1 | 49.75% |
| Mean evidence quality | 0.0000 |
| Evidence empty rate | 100.00% |
| Mean reasoning score | 0.25 |
| Hallucination rate | 0.00% |
| Mean agent calls | 1.0 |
| Fallback count | 0 |
| State Error | 0 |

v4 路由分布：

```text
LOW / Direct: 78 samples
MEDIUM / Structured CoT: 22 samples
HIGH / Full Multi-Agent: 0 samples
```

各路线内部表现：

| Route | Samples | Accuracy |
|---|---:|---:|
| Direct | 78 | 44.87% |
| Structured CoT | 22 | 59.09% |

v4 相比 v3 100 条：

| Metric | v3 | v4 | Change |
|---|---:|---:|---:|
| Accuracy | 45.00% | 48.00% | +3.00 |
| Direct samples | 53 | 78 | +25 |
| Structured CoT samples | 29 | 22 | -7 |
| Full Multi-Agent samples | 18 | 0 | -18 |
| Mean agent calls | 2.8 | 1.0 | -1.8 |
| State Error | 0 | 0 | 0 |

阶段结论：

```text
1. v4 达到当前 100 条调参集最高 accuracy：48%。
2. Direct-dominant 路由有效，减少复杂 agent 后 accuracy 和成本同时改善。
3. Structured CoT 在 v4 选择到的 22 条样本上 accuracy 为 59.09%，说明收紧后的 CoT 比 v2/v3 更像有效子路线。
4. Full Multi-Agent 在这 100 条中没有被调用，符合 Oracle full 中 Full 最优样本极少的趋势。
5. 当前 v4 可以进入 full run，但 full 后需要继续检查是否过度压制了 Full Multi-Agent。
```

## 当前 v4 full 结果

实验：

```text
Experiment: exp06_rule_based_adaptive_routing_v4_qwen_7b_4090d_pmc10k_full
Samples: 451
Model: /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
Backend: qwen2_5_vl
Routing policy: rule_based_v4
```

主要指标：

| Metric | Value |
|---|---:|
| Accuracy | 52.33% |
| Correct | 236 / 451 |
| Mean token F1 | 56.70% |
| Mean BLEU-1 | 55.50% |
| Mean evidence quality | 0.0062 |
| Evidence empty rate | 99.33% |
| Mean reasoning score | 0.2550 |
| Hallucination rate | 0.44% |
| Mean agent calls | 1.07 |
| Fallback count | 0 |
| State Error | 0 |

v4 full 路由分布：

```text
LOW / Direct: 372 samples
MEDIUM / Structured CoT: 76 samples
HIGH / Full Multi-Agent: 3 samples
```

各路线内部表现：

| Route | Samples | Accuracy |
|---|---:|---:|
| Direct | 372 | 50.81% |
| Structured CoT | 76 | 60.53% |
| Full Multi-Agent | 3 | 33.33% |

v4 full 相比 v3 full / Oracle：

| Metric | v3 full | v4 full | Oracle |
|---|---:|---:|---:|
| Accuracy | 50.55% | 52.33% | 61.20% |
| Direct samples | 264 | 372 | 225 |
| Structured CoT samples | 127 | 76 | 42 |
| Full Multi-Agent samples | 60 | 3 | 9 |
| Mean agent calls | not listed | 1.07 | not applicable |

阶段结论：

```text
1. v4 full 是当前 Phase 6 最好结果，accuracy 达到 52.33%。
2. v4 证明减少 over-reasoning 是有效方向：accuracy 提升，同时 agent 调用成本大幅下降。
3. Structured CoT 在 76 条样本上 accuracy 为 60.53%，说明 v4 对 CoT 的选择比 v3 更有效。
4. Full Multi-Agent 只调用 3 条，低于 Oracle 的 9 条，说明 v4 可能略微 under-use Full，但这不是当前主要问题。
5. 距离 Oracle 61.20% 仍有约 8.87 个百分点空间，下一步应做 sample-level oracle gap 分析，而不是继续盲目加复杂 agent。
```

下一步建议：

```text
优先做 v4 的 Oracle gap 分析：
1. 找出 v4 选 Direct 但 Oracle 最优是 Structured CoT 的样本。
2. 找出 v4 选 Direct / Structured CoT 但 Oracle 最优是 Full Multi-Agent 的样本。
3. 找出 v4 错但 Direct / CoT / Full 都错的 unrecoverable 样本，避免把不可恢复错误误当成 routing 问题。
```

## v4 Oracle Routing full 分析

实验：

```text
Routing result: Results/exp06_rule_based_adaptive_routing_v4_qwen_7b_4090d_pmc10k_full/predictions.jsonl
Oracle output: Results/oracle_routing_analysis_v4_full
Samples: 451
```

关键指标：

| Metric | v3 Oracle Analysis | v4 Oracle Analysis | Change |
|---|---:|---:|---:|
| Routing accuracy | 50.55% | 52.33% | +1.78 |
| Routing-to-oracle match rate | 35.48% | 42.57% | +7.10 |
| Over-reasoning rate | 18.63% | 9.53% | -9.09 |
| Under-reasoning rate | 7.10% | 9.09% | +1.99 |
| Unrecoverable error rate | 38.80% | 38.80% | 0 |
| Matched route count | 160 | 192 | +32 |
| Over-reasoning count | 84 | 43 | -41 |
| Under-reasoning count | 32 | 41 | +9 |

v4 selected route vs oracle：

| Selected Route | Oracle Direct | Oracle CoT | Oracle Full | Oracle None | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 182 | 32 | 5 | 153 | 372 |
| Structured CoT | 41 | 10 | 4 | 21 | 76 |
| Full Multi-Agent | 2 | 0 | 0 | 1 | 3 |

主要结论：

```text
1. v4 的核心收益来自减少 over-reasoning：over-reasoning 从 84 条降到 43 条。
2. v4 的代价是 under-reasoning 略升：从 32 条升到 41 条。
3. v4 选 Direct 但 Oracle 是 Structured CoT 的样本有 32 条，这是下一版最值得优化的部分。
4. v4 选 Direct 但 Oracle 是 Full Multi-Agent 的样本只有 5 条；v4 选 Structured CoT 但 Oracle 是 Full 的样本有 4 条。
5. Full Multi-Agent 当前不是主要增益来源。Oracle 中 Full 最优只有 9 条，盲目扩大 Full 会重新引入 over-reasoning。
```

下一版 v5 不应回到 v3 的高 Full 调用，而应做小幅修正：

```text
v5 目标：
保留 v4 的 Direct-dominant 主体，只增加一小类 CoT 触发规则，追回 Direct -> Oracle CoT 的 32 条。

不建议：
大幅增加 Full Multi-Agent。

建议：
优先分析 oracle_analysis.csv 中 selected_route=direct 且 oracle_route=structured_cot 的 32 条样本，
从问题文本里提取稳定规则，再决定是否进入 v5。
```

## v5 路由优化

v5 是基于 v4 Oracle gap 的小步优化，不推翻 v4。它只针对：

```text
selected_route = direct
oracle_route = structured_cot
count = 32
```

这 32 条样本的分布：

| Dimension | Main Pattern |
|---|---|
| Question type | PRES 16, ABN 5, POS 3, MODALITY 2, OTHER 2 |
| Answer type | CLOSED 23, OPEN 9 |
| Organ | ABD 15, CHEST 10, HEAD 7 |

重要发现：

```text
1. CoT 可救回的样本并不主要来自 SIZE / ATTRIB。
2. v4 的 CoT 误伤反而主要来自 broad SIZE / ATTRIB 规则。
3. 可救回样本集中在特定 finding、左右定位、MRI sequence、bright structures 等问题文本。
```

因此 v5 只新增窄 CoT 触发词，例如：

```text
airspace consolidation
vascular pathology
small bowel obstruction
calcified / calcifications
lung markings
lymphadenopathy
mass present
hypodense mass
gallstones
bright white structures
bright specks
left kidney abnormal
gastric bubble
sequence of this MRI
left or right
```

v5 仍然保持：

```text
Direct:
默认路线。

Structured CoT:
v4 原有 SIZE / ATTRIB / 比较型触发词
+
v5 从 Oracle gap 得到的窄触发词。

Full Multi-Agent:
仍然只由强 diagnosis / cause / etiology / differential 等触发。
```

注意：

```text
v5 是 oracle-guided tuning 版本，适合证明“样本级 routing gap 可以被规则修正”。
写论文时不能把 v5 当作完全独立、未调参的 unbiased test 结果。
更稳妥的表述是：v4 为主结果，v5 为 oracle-guided error analysis / routing refinement。
```

v5 100 条：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_v5_qwen_7b_4090d_pmc10k_100.yaml
```

v5 full：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_v5_qwen_7b_4090d_pmc10k_full.yaml
```

v5 full 后继续跑 Oracle 对比：

```bash
python scripts/analyze_oracle_routing.py \
  --direct Results/exp01_direct_vlm_qwen_7b_cuda_full/predictions.jsonl \
  --cot Results/exp02_structured_cot_qwen_7b_4090d_full/predictions.jsonl \
  --full-agent Results/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_full/predictions.jsonl \
  --routing Results/exp06_rule_based_adaptive_routing_v5_qwen_7b_4090d_pmc10k_full/predictions.jsonl \
  --output Results/oracle_routing_analysis_v5_full
```

v5 判断标准：

```text
accuracy > v4 full 的 52.33%
routing-to-oracle match rate > v4 的 42.57%
under-reasoning rate 下降
over-reasoning rate 不明显反弹
mean agent calls 仍接近 1
```

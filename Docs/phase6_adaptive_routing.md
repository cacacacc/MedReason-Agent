# Phase 6：Supervisor Adaptive Routing

Phase 6 是项目的核心阶段。前面 Phase 4 / Phase 5 的结果说明：固定地让所有样本走完整
Multi-Agent、RAG、Verifier，并不会稳定提升 VQA-RAD accuracy。相反，简单 closed visual
questions 往往会被过度推理和无关 evidence 干扰。

因此 Phase 6 的目标是：

```text
不让所有问题都跑完整 Multi-Agent，而是根据难度和不确定性动态选择 reasoning depth。
```

## 第一版路线

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

每条样本会记录：

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

规则如下：

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
时才触发 fallback。这样避免把大量本来正确的短答案再次送入复杂链路。

## 指标

Phase 6 除了继承 accuracy、token F1、BLEU-1、evidence quality、claim status、agent
metrics 外，还新增：

```text
route_distribution
complexity_distribution
fallback_rate
fallback_count
route_accuracy
mean_agent_calls
```

## 实验指令

先跑 Phase 4 heuristic full，作为当前最强 Multi-Agent 候选：

```bash
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_full.yaml
```

再跑 Phase 6 100 条：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_qwen_7b_4090d_pmc10k_100.yaml
```

如果 100 条满足下面条件，再跑 full：

```text
accuracy >= heuristic routing 100 条结果
route_distribution 不是全部 HIGH
mean_agent_calls 明显低于完整 Multi-Agent
fallback_rate 不过高
```

full 指令：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_qwen_7b_4090d_pmc10k_full.yaml
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

## v2 路由优化

100 条 v1 tuning 结果显示：

```text
Direct route: 39 samples, 56.41%
Structured CoT route: 13 samples, 76.92%
Full Multi-Agent route: 48 samples, 22.92%
```

这说明 v1 的主要问题不是 adaptive routing 方向错，而是 HIGH 路由太多。尤其是 v1 把
所有 OPEN question 都直接送入 Full Multi-Agent，导致完整 agent 链路拖低整体 accuracy。

v2 做了三点修改：

```text
1. 不再把所有 OPEN question 直接判为 HIGH。
2. OPEN + visual description / anatomy / location / abnormality description -> MEDIUM。
3. 只有 diagnosis / disease / cause / etiology / differential / compatible / consistent with
   这类强医学知识触发词才进入 HIGH。
```

v2 的目标路由分布是：

```text
LOW: 35%-45%
MEDIUM: 25%-40%
HIGH: 20%-30%
```

同时，v2 修复了 Direct / Structured CoT 路线的最小 shared state 记录。它们不使用完整
Multi-Agent shared state，但会写入兼容字段，避免 error attribution 把正常的 Direct/CoT
路线误判为 State Error。

v2 100 条指令：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_v2_qwen_7b_4090d_pmc10k_100.yaml
```

v2 full 指令：

```bash
python scripts/run_phase6_adaptive_routing.py --config configs/experiments/exp06_rule_based_adaptive_routing_v2_qwen_7b_4090d_pmc10k_full.yaml
```

v2 是否进入 full 的判断标准：

```text
accuracy >= 43%
HIGH route 占比低于 v1
Full Multi-Agent route accuracy 不再明显拖低整体结果
State Error 明显下降
mean_agent_calls 不高于 v1
```

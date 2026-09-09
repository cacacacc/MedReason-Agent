# Phase 4 Supervisor Multi-Agent Results

## 实验设置

本阶段比较 Fixed Multi-Agent 和 Supervisor Multi-Agent，模型、数据集和测试集保持一致。

```text
模型：Qwen/Qwen2.5-VL-7B-Instruct
本地模型路径：/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
Backend：qwen2_5_vl
任务：VQA-RAD Medical VQA
数据版本：OSF public release
Split：test
Full test：451 samples
训练 epochs：0
Backbone：frozen
```

Supervisor 的 RAG 实验使用：

```text
Corpus：PMC 10k abstracts
Retriever：FAISS
Embedding：/root/autodl-tmp/models/bge-small-en-v1.5
Reranker：keyword_reranker
Candidate top-k：10
Evidence top-k：5
```

## 实验路径

Fixed Multi-Agent：

```text
Vision Agent -> Reasoning Agent -> Critic Agent -> Answer Agent
```

Supervisor Multi-Agent：

```text
Supervisor Agent -> Vision Agent -> Retrieval Agent
-> Reasoning Agent -> Verifier Agent -> Answer Agent
```

## 主结果

Full test 结果如下：

| Method | Samples | Accuracy | Correct | Token F1 | BLEU-1 | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| Direct VLM | 451 | 49.89% | 225 / 451 | 56.09% | 54.59% | 4.36 s* |
| CoT Prompt | 451 | 43.90% | 198 / 451 | 51.24% | 49.52% | 72.41 s* |
| Fixed Multi-Agent | 451 | 39.69% | 179 / 451 | 42.15% | 41.61% | 9.32 s |
| Supervisor Multi-Agent + PMC 10k | 451 | 31.26% | 141 / 451 | 37.31% | 35.71% | 14.86 s |

`*` Direct VLM 和 CoT 的 latency 沿用此前运行记录；本次更新的 Phase 4 latency 来自最新
预测文件的 `latency_ms` 统计。

相对 Direct VLM：

```text
Fixed Multi-Agent delta：-10.20 percentage points
Supervisor Multi-Agent delta：-18.63 percentage points
```

相对 CoT Prompt：

```text
Fixed Multi-Agent delta：-4.21 percentage points
Supervisor Multi-Agent delta：-12.64 percentage points
```

当前结果表明，简单拆分角色并没有超过单模型 Direct VLM；加入 Supervisor 和检索验证后，
证据指标有所改善，但最终答案准确率进一步下降。

## Fixed Multi-Agent Full Test

```text
Experiment：exp04_fixed_multi_agent_qwen_7b_4090d_full
Accuracy：0.3968957871396896
Correct：179 / 451
Mean token F1：0.4214566653004112
Mean BLEU-1：0.4160617885518791
Mean latency：9316.02 ms
Median latency：9452.40 ms
P90 latency：11534.72 ms
Max latency：18240.64 ms
```

按答案类型：

```text
CLOSED：171 / 272 = 62.87%
OPEN：8 / 179 = 4.47%
```

Fixed Multi-Agent 没有启用外部检索，因此：

```text
Evidence empty rate：1.0
Mean evidence quality：0.0
Claim status：911 HYPOTHESIS
Critic ACCEPT：248
Critic REVISE：202
```

当前记录中的 `HYPOTHESIS` 是 reasoning claims 的默认状态，不等同于全部 claim 都被判定为
错误。该路径的 hallucination rate 为 `0.5122`，但没有 retrieval evidence 可用于验证 claim。

## Supervisor Multi-Agent Full Test

```text
Experiment：exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_full
Accuracy：0.31263858093126384
Correct：141 / 451
Mean token F1：0.3731050600197594
Mean BLEU-1：0.3571008839408528
Mean latency：14858.46 ms
Median latency：14769.46 ms
P90 latency：18232.35 ms
Max latency：23809.64 ms
```

按答案类型：

```text
CLOSED：113 / 272 = 41.54%
OPEN：14 / 179 = 7.82%
```

Evidence 指标：

```text
Mean evidence quality：0.690401199556541
Mean question coverage：0.7291239623059868
Mean answer coverage：0.501041665
Evidence empty rate：0.0
Retrieved evidence：5 条 / sample
```

Claim status 总计：

```text
SUPPORTED：421
UNSUPPORTED：646
HYPOTHESIS：1067
CONTRADICTED：0
OBSERVED：0
```

Supervisor 的 hallucination rate 为 `0.6208`，高于 Fixed Multi-Agent 的 `0.5122`。
这说明当前 Verifier 虽然能够产生 claim-level labels，但并没有稳定地把不支持的 claim
转化为更准确的最终答案。

## Error Attribution

Full Supervisor 结果经过 `backfill_error_attribution.py` 处理后：

```text
Incorrect samples：324
Routing Error：324
Human review required：324
Unknown error：0
```

候选错误类型统计为：

```text
Perception Error：324
Retrieval Error：12
Verification Error：111
Reasoning Error：0
Routing Error：324
State Error：0
```

这里的 `Routing Error` 是当前自动归因规则的结果，不应直接解释为 324 个样本都已经被
人工确认是路由错误。由于当前 Supervisor 的 `expected_selected_tools` 与实际输出存在
格式和选择差异，自动归因把所有 incorrect samples 都标记为 routing-related，并保留
`human_review_error_count=324`。正式论文结果应在人工抽样复核后再报告细分类别。

## Supervisor Smoke Test

```text
Experiment：exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_20
Samples：20
Accuracy：0.30
Correct：6 / 20
Mean token F1：0.31825396825396823
Mean BLEU-1：0.31009615384615385
Mean evidence quality：0.71166665
Mean question coverage：0.72166665
Evidence empty rate：0.0
Hallucination rate：0.55
```

Claim status：

```text
SUPPORTED：20
HYPOTHESIS：46
UNSUPPORTED：26
CONTRADICTED：0
```

Smoke 结果的 accuracy 高于 full test，但 20 条样本不足以支持稳定结论。它主要用于确认
Supervisor、FAISS、Qwen 本地模型和结果写入流程正常。

## 与 Phase 3 的关系

在同一 PMC 10k corpus 上：

```text
Knowledge RAG accuracy：22.84%
Knowledge + Evidence RAG accuracy：11.53%
Fixed Multi-Agent accuracy：39.69%
Supervisor Multi-Agent accuracy：31.26%
```

Fixed Multi-Agent 高于两条 Phase 3 RAG 路径，但仍低于 Direct VLM。Supervisor Multi-Agent
的 evidence quality 高于简单 Knowledge RAG，但最终答案 accuracy 低于 Fixed Multi-Agent
和 Direct VLM。

## 阶段结论

当前 Phase 4 full test 的主要结论是：

1. Fixed Multi-Agent 没有超过 Direct VLM，accuracy 从 `49.89%` 降至 `39.69%`。
2. Supervisor Multi-Agent 没有超过 Fixed Multi-Agent，accuracy 为 `31.26%`。
3. Supervisor 的 evidence quality 为 `0.6904`，但较好的证据覆盖没有转化为更好的最终答案。
4. Supervisor 的 hallucination rate 为 `0.6208`，说明当前 claim verification 和 answer
   generation 之间仍存在明显断裂。
5. CLOSED 问题明显优于 OPEN 问题，两个 Multi-Agent 路径都存在严重的开放式答案困难。
6. 当前自动 error attribution 对 routing 的归因过于集中，必须经过人工抽样后才能用于正式错误分析。

因此，Phase 4 当前更适合作为 multi-agent pipeline 和审计字段的验证结果，而不是证明
Supervisor 架构提升了 VQA accuracy 的正向结果。

## 下一步

1. 修正 Supervisor 的 `expected_selected_tools` 和实际输出格式，重新计算 tool selection accuracy。
2. 让 Answer Agent 对 `UNSUPPORTED` claim 进行短答案约束，而不是直接生成冗长或过度谨慎的答案。
3. 增加 deterministic answer gate，并单独报告 gate enabled / disabled 的结果。
4. 对最新 Supervisor full 的 310 个 incorrect samples 做人工抽样，校准 Perception、Routing 和 Verification 的归因规则。
5. 将 Dynamic Gate 配置作为 Phase 5 独立实验，不与当前 Phase 4 full 结果混合。

## 100-Sample Tuning / Ablation Results

以下四组实验均使用 VQA-RAD test split 的前 100 条样本，属于 tuning / ablation 结果，
不与 451 条 full test 结果混合。四组预测文件均为 100/100 条，唯一 sample_id 为 100，
空预测为 0。

| Method | Top-k | Accuracy | Correct | Token F1 | BLEU-1 | Evidence Quality | Empty Evidence | Tool Selection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Supervisor baseline | 5 | 35% | 35 / 100 | 36.90% | 36.65% | 0.6897 | 0.0 | 0.0 |
| Supervisor top-k ablation | 3 | 31% | 31 / 100 | 32.28% | 31.85% | 0.6027 | 0.0 | 0.0 |
| Supervisor top-k ablation | 10 | 36% | 36 / 100 | 37.65% | 37.30% | 0.7567 | 0.0 | 0.0 |
| Heuristic question routing | 5 | 41% | 41 / 100 | 42.71% | 42.35% | 0.0778 | 0.87 | 1.0 |

### Top-k Ablation

在默认 Supervisor 路由下，top-k=10 的 accuracy 为 `36%`，高于 top-k=5 的 `35%` 和
top-k=3 的 `31%`。top-k=10 同时获得最高 evidence quality、question coverage 和
answer coverage：

```text
top-k=3：evidence quality 0.6027，question coverage 0.6333，answer coverage 0.3059
top-k=5：evidence quality 0.6897，question coverage 0.7144，answer coverage 0.4382
top-k=10：evidence quality 0.7567，question coverage 0.7667，answer coverage 0.5961
```

这组 100 条结果显示，增加 evidence 数量改善了规则版证据覆盖，但 accuracy 提升有限，
且 top-k=10 的 hallucination rate `0.39` 低于 top-k=5 的 `0.61` 和 top-k=3 的 `0.77`。

### Heuristic Question Routing

Heuristic routing 获得了四组中最高的 accuracy：

```text
Accuracy：0.41
Correct：41 / 100
Mean reasoning score：0.805
Hallucination rate：0.04
Tool selection accuracy：1.0
Evidence empty rate：0.87
```

它的主要代价是大量样本没有检索证据，evidence empty rate 达到 `0.87`，mean evidence
quality 只有 `0.0778`。因此这组结果不能简单解释为“检索质量更好”，更可能说明启发式
路由在部分问题上跳过了检索并减少了错误传播。它需要在更大样本和统一 latency 统计下
继续验证。

### 100-Sample 结论

```text
最高 accuracy：Heuristic routing，41%
最高 evidence quality：top-k=10，0.7567
最低 hallucination rate：Heuristic routing，0.04
```

这四组 100 条实验适合用于选择下一轮 full run 的候选配置，但不应替代 451 条 full test
结果。下一步应优先验证 heuristic routing 和 top-k=10，并补充严格一致的运行成本指标。

## Additional 100-Sample Memory / Routing Ablations

随后完成的三组 100 条实验同样属于 tuning / ablation，不与 451 条 full test 结果混合。
三组预测均为 100/100 条，唯一 sample_id 为 100，空预测为 0。

| Method | Accuracy | Token F1 | BLEU-1 | Evidence Quality | Empty Evidence | Hallucination | Tool Selection | Memory Hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heuristic + k10 | 39% | 40.71% | 40.35% | 0.0862 | 87% | 3% | 1.0 | 0.00 |
| Supervisor + Memory | 34% | 37.48% | 36.39% | 0.6712 | 0% | 41% | 0.0 | 2.93 |
| Heuristic + k10 + Memory | 38% | 41.76% | 40.98% | 0.0887 | 87% | 1% | 1.0 | 2.93 |

### 结果解读

```text
Heuristic + k10：accuracy 39%，tool selection 1.0，但 evidence empty rate 0.87。
Supervisor + Memory：evidence quality 0.6712，memory hits 2.93，但 accuracy 34%。
Heuristic + k10 + Memory：accuracy 38%，hallucination 1%，但 evidence empty rate 0.87。
```

Memory 在 heuristic routing 下将 hallucination 从 `3%` 降到 `1%`，但 accuracy 从 `39%`
降到 `38%`；在普通 Supervisor 路由下，memory 提供了平均 `2.93` 条历史记忆，但 accuracy
只有 `34%`。这说明 memory 当前更像可靠性审计辅助信号，还没有稳定转化为答案质量提升。

与此前 heuristic + k10 的 100 条结果相比，本次结果的 accuracy 为 `39%`，低于此前的
`41%`；由于两次实验均为 100 条 tuning 样本，不能据此下 full-test 结论。

新增结果文件：

```text
Results/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_k10_100/predictions.jsonl
Results/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_k10_100/metrics.json
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_memory_100/predictions.jsonl
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_memory_100/metrics.json
Results/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_k10_memory_100/predictions.jsonl
Results/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_k10_memory_100/metrics.json
```

## 结果文件

```text
Results/exp04_fixed_multi_agent_qwen_7b_4090d_full/predictions.jsonl
Results/exp04_fixed_multi_agent_qwen_7b_4090d_full/metrics.json
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_full/predictions.jsonl
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_full/metrics.json
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_20/predictions.jsonl
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_20/metrics.json
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_100/predictions.jsonl
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_100/metrics.json
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_k3_100/predictions.jsonl
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_k3_100/metrics.json
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_k10_100/predictions.jsonl
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_k10_100/metrics.json
Results/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_100/predictions.jsonl
Results/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_100/metrics.json
```

`Results/` 被 `.gitignore` 忽略，因此本文件作为 Phase 4 实验结果的 tracked summary。
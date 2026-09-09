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
| Direct VLM | 451 | 49.22% | 222 / 451 | 55.41% | 53.97% | 4.36 s |
| CoT Prompt | 451 | 43.02% | 194 / 451 | 51.19% | 49.24% | 72.41 s |
| Fixed Multi-Agent | 451 | 39.02% | 176 / 451 | 45.63% | 44.13% | 11.61 s |
| Supervisor Multi-Agent + PMC 10k | 451 | 28.16% | 127 / 451 | 35.22% | 33.24% | 14.86 s |

相对 Direct VLM：

```text
Fixed Multi-Agent delta：-10.20 percentage points
Supervisor Multi-Agent delta：-21.06 percentage points
```

相对 CoT Prompt：

```text
Fixed Multi-Agent delta：-3.99 percentage points
Supervisor Multi-Agent delta：-14.86 percentage points
```

当前结果表明，简单拆分角色并没有超过单模型 Direct VLM；加入 Supervisor 和检索验证后，
证据指标有所改善，但最终答案准确率进一步下降。

## Fixed Multi-Agent Full Test

```text
Experiment：exp04_fixed_multi_agent_qwen_7b_4090d_full
Accuracy：0.3902439024390244
Correct：176 / 451
Mean token F1：0.45631439448773475
Mean BLEU-1：0.44125806397068107
Mean latency：11612.85 ms
Median latency：11602.75 ms
P90 latency：14257.76 ms
Max latency：25942.85 ms
```

按答案类型：

```text
CLOSED：156 / 272 = 57.35%
OPEN：20 / 179 = 11.17%
```

Fixed Multi-Agent 没有启用外部检索，因此：

```text
Evidence empty rate：1.0
Mean evidence quality：0.0
Claim status：941 HYPOTHESIS
Critic ACCEPT：248
Critic REVISE：202
```

当前记录中的 `HYPOTHESIS` 是 reasoning claims 的默认状态，不等同于全部 claim 都被判定为
错误。该路径的 hallucination rate 为 `0.4523`，但没有 retrieval evidence 可用于验证 claim。

## Supervisor Multi-Agent Full Test

```text
Experiment：exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_full
Accuracy：0.28159645232815966
Correct：127 / 451
Mean token F1：0.35219225762136935
Mean BLEU-1：0.33242786345340886
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

Supervisor 的 hallucination rate 为 `0.6208`，高于 Fixed Multi-Agent 的 `0.4523`。
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
Fixed Multi-Agent accuracy：39.02%
Supervisor Multi-Agent accuracy：28.16%
```

Fixed Multi-Agent 高于两条 Phase 3 RAG 路径，但仍低于 Direct VLM。Supervisor Multi-Agent
的 evidence quality 高于简单 Knowledge RAG，但最终答案 accuracy 低于 Fixed Multi-Agent
和 Direct VLM。

## 阶段结论

当前 Phase 4 full test 的主要结论是：

1. Fixed Multi-Agent 没有超过 Direct VLM，accuracy 从 `49.22%` 降至 `39.02%`。
2. Supervisor Multi-Agent 没有超过 Fixed Multi-Agent，accuracy 为 `28.16%`。
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
4. 对 324 个 incorrect samples 做人工抽样，校准 Perception、Routing 和 Verification 的归因规则。
5. 将 Dynamic Gate 配置作为 Phase 5 独立实验，不与当前 Phase 4 full 结果混合。

## 结果文件

```text
Results/exp04_fixed_multi_agent_qwen_7b_4090d_full/predictions.jsonl
Results/exp04_fixed_multi_agent_qwen_7b_4090d_full/metrics.json
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_full/predictions.jsonl
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_full/metrics.json
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_20/predictions.jsonl
Results/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_20/metrics.json
```

`Results/` 被 `.gitignore` 忽略，因此本文件作为 Phase 4 实验结果的 tracked summary。
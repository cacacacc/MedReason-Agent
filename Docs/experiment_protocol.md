# 实验协议

所有主实验都必须遵守本协议，除非某个实验明确研究其中一个变量。这个项目的实验目标是做 controlled comparison，也就是一次只改变一个关键因素，尽量避免“结果变好了但不知道为什么”的问题。

## 固定变量

- Backbone：主实验使用同一个 VLM。
- Dataset split：同一个数据集使用固定划分。
- Prompt version：提示词必须版本化，并写入结果记录。
- Image preprocessing：图像预处理必须固定，并写入实验配置。
- Generation settings：尽可能使用确定性解码。
- Evaluation scripts：可比较实验必须使用同一套评估脚本。

## 默认模型策略

主实验默认：

- Qwen2.5-VL-7B-Instruct

轻量开发 fallback：

- Qwen2.5-VL-3B-Instruct

外部 baseline：

- LLaVA-Med

不要在主对照链里混用不同 backbone。否则无法判断性能变化来自 reasoning architecture，还是来自模型本身不同。

## 默认生成参数

```yaml
temperature: 0.0
top_p: 1.0
max_new_tokens: 1024
seeds:
  - 42
  - 2024
  - 3407
```

如果某个 API 或模型服务不支持严格 seed 控制，必须在 run metadata 里记录这个限制。

## 数据规模策略

每个新 pipeline 都必须按下面顺序扩大规模：

```text
Smoke Test: 10-20 samples
Dev Experiment: 100-200 samples
Intermediate Experiment: 500-1000 samples
Full Experiment: 只有 pipeline 稳定后才运行
```

这样做是为了避免一开始就在错误 pipeline 上浪费算力。

## 主实验链

```text
Experiment 0: Random / Majority Baseline
Experiment 1: Direct VLM
Experiment 2: Structured Direct Prompt
Experiment 3: Chain-of-Thought
Experiment 4: Self-Consistency on small subset
Experiment 5: RAG-Augmented Reasoning
Experiment 6: Fixed Multi-Agent
Experiment 7: Supervisor Multi-Agent
Experiment 8: Supervisor + Verification
Experiment 9: Adaptive Reasoning
Experiment 10: Optional LoRA
```

## 必须记录的结果字段

每条 prediction record 至少包含：

```json
{
  "experiment_id": "",
  "model": "",
  "prompt_version": "",
  "prompt_contract": "",
  "rag_mode": "",
  "agent_mode": "",
  "dataset": "",
  "split": "",
  "sample_id": "",
  "image_id": "",
  "question": "",
  "prediction": "",
  "ground_truth": "",
  "reasoning_output": "",
  "agent_outputs": {},
  "shared_state": {},
  "state_compression": {},
  "retrieved_evidence": [],
  "knowledge_query": "",
  "evidence_query": "",
  "knowledge_evidence": [],
  "verification_evidence": [],
  "generated_claims": [],
  "claim_statuses": [
    {
      "claim": "",
      "status": "OBSERVED | SUPPORTED | HYPOTHESIS | UNSUPPORTED | CONTRADICTED"
    }
  ],
  "initial_prediction": "",
  "initial_reasoning_output": "",
  "verified_claim": "",
  "claim_verification_status": "",
  "evidence_quality_score": null,
  "evidence_quality": {},
  "tool_calls": [],
  "selected_tools": [],
  "expected_selected_tools": [],
  "agent_route": [],
  "expected_agent_route": [],
  "critic_decision": "",
  "confidence": null,
  "latency_ms": null,
  "input_tokens": null,
  "output_tokens": null,
  "correct": null,
  "error_type": ""
}
```

## Claim Status 协议

Phase 3 及之后的所有带 reasoning / verifier 的实验，都必须记录 `claim_statuses`。
每条医学信息都要绑定一个状态：

```text
OBSERVED: 图像中可直接观察到的 finding。
SUPPORTED: 已被 retrieved evidence 或 verifier 支持的 claim。
HYPOTHESIS: 合理但尚未验证的候选判断。
UNSUPPORTED: 当前图像或证据不足以支持的 claim。
CONTRADICTED: 与 retrieved evidence 冲突的 claim。
```

关键规则：

```text
HYPOTHESIS 不能在分析中当作 fact。
只有 OBSERVED 和 SUPPORTED 可以作为较强证据写入结论。
UNSUPPORTED / CONTRADICTED 会进入 hallucination 相关分析。
```

## Shared State / State Compression 协议

Phase 4 及之后的 Multi-Agent 实验使用 sample-level shared state。这个 memory 只在
单条样本内部有效，不跨病例保存，避免训练集或测试集之间的信息泄漏。

每个 agent 的通信流程：

```text
Agent raw output
-> State Compression
-> SharedAgentState
-> Next Agent reads compressed_context
```

结果记录必须保存：

```text
shared_state: 完整共享状态审计记录
state_compression: 压缩方法、原始字符数、压缩字符数、compression_ratio
memory_records: 从磁盘持久 memory 检索到的历史经验
memory_write_record: 当前样本写入磁盘的 memory record
```

当前压缩方法为 `section_extract_v1`，只保留关键结构化段落和 claim statuses。

Persistent Memory 是跨运行、跨对话存在的磁盘 memory。默认实现为 JSONL：

```yaml
memory:
  enabled: true
  path: Experiments/memory/phase4_supervisor_qwen_4090d_memory.jsonl
  namespace: phase4_supervisor_qwen_4090d
  top_k: 3
  min_score: 0.05
  read_enabled: true
  write_enabled: true
  include_prediction: false
```

主实验默认不启用 persistent memory。启用它时必须作为单独 ablation 报告，因为它会让
后续样本读取早先样本的经验。为了降低泄漏风险，当前 memory record 默认：

```text
不保存 ground_truth
不把 prediction 写入 memory_text
只保存 question metadata、claim_statuses、evidence titles 和压缩经验
```

## 报告要求

最终报告需要包含：

- 多次运行时的 mean ± std。
- Accuracy 和任务相关答案指标。
- Reliability metrics。
- Efficiency metrics。
- Error taxonomy breakdown。

不要只把结果打印到 terminal。所有实验结果必须保存成结构化 JSON / JSONL / CSV。

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
  "dataset": "",
  "split": "",
  "sample_id": "",
  "image_id": "",
  "question": "",
  "prediction": "",
  "ground_truth": "",
  "reasoning_output": "",
  "retrieved_evidence": [],
  "tool_calls": [],
  "agent_route": [],
  "critic_decision": "",
  "confidence": null,
  "latency_ms": null,
  "input_tokens": null,
  "output_tokens": null,
  "correct": null,
  "error_type": ""
}
```

## 报告要求

最终报告需要包含：

- 多次运行时的 mean ± std。
- Accuracy 和任务相关答案指标。
- Reliability metrics。
- Efficiency metrics。
- Error taxonomy breakdown。

不要只把结果打印到 terminal。所有实验结果必须保存成结构化 JSON / JSONL / CSV。

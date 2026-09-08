# Phase 4：Supervisor Multi-Agent

## Phase Goal

Phase 4 是项目核心阶段，目标是证明：

```text
Agent collaboration 有价值。
```

更具体地说，本阶段要比较单模型直接回答、CoT、固定 Multi-Agent 和
Supervisor Multi-Agent 在医学多模态 VQA 上的可靠性差异。

## 数据口径

当前仓库使用 OSF 公共版 VQA-RAD：

```text
Raw QA records: 2248
Full test: 451
```

因此本仓库 Phase 4 full test 使用 451 条 VQA-RAD test samples，不写 464。
如果后续切换到包含 3,000 QA / 464 test 的 VQA-RAD 镜像，必须新增数据导入脚本和
版本审计文档，不能和当前 OSF 结果混在一起。

PathVQA 作为后续扩展数据集：

```text
PathVQA: 5,000-10,000 QA subset
```

但当前 Phase 4 第一版只实现 VQA-RAD。

## 对照实验

```text
Baseline 1: Single VLM
Baseline 2: VLM + CoT
Baseline 3: Fixed Multi-Agent
Method: Supervisor Multi-Agent
```

所有方法使用同一个 backbone：

```text
Qwen2.5-VL-7B-Instruct
training_epochs: 0
frozen_backbone: true
```

## Fixed Multi-Agent

固定链路：

```text
Vision Agent
-> Reasoning Agent
-> Critic Agent
-> Answer Agent
```

它不做动态工具选择，因此用于回答一个问题：

```text
把单模型拆成多个角色 prompt，是否比 CoT 更可靠？
```

## Supervisor Multi-Agent

Supervisor 链路：

```text
Supervisor Agent
-> Vision Agent
-> Retrieval Agent
-> Reasoning Agent
-> Verifier Agent
-> Answer Agent
```

Supervisor 负责选择工具，Retrieval Agent 使用 Phase 3 的 RAG 检索链路，
Verifier Agent 检查 reasoning claims 是否被 evidence 支持。

## Memory / Shared State

当前 Phase 4 加入的是 sample-level short-term memory，不是跨病例长期记忆：

```text
SharedAgentState
-> selected_tools
-> compressed agent memory
-> retrieved_evidence
-> claim_statuses
-> compressed_context
```

每个 agent 不再只读取上一轮 raw output，而是读取压缩后的 shared state：

```text
Agent writes raw output
-> State Compression extracts key sections
-> SharedAgentState updates claims / evidence / tool choices
-> Next Agent reads compressed_context
```

State Compression 当前使用规则版 `section_extract_v1`，只抽取关键段落，例如
`Observation`、`Reasoning`、`Claims`、`Claim Statuses`、`Verification Status`、
`Final Answer`。这样能保留可审计信息，同时减少 prompt 长度，避免把未验证的长推理
原文直接传给下游 agent。

## Persistent Memory

如果需要“跨对话也存在”的真正 memory，可以启用 persistent memory。它会写入磁盘
JSONL，下次运行或下次对话仍然能读取：

```text
Previous run writes memory JSONL
-> New run retrieves similar memories
-> SharedAgentState includes Persistent Memory
-> Agents read it through compressed_context
```

当前实现文件：

```text
src/medreason_agent/agents/memory.py
```

默认安全策略：

```text
stores_ground_truth: false
include_prediction: false
```

也就是说，memory 默认不保存标准答案，也不把历史模型答案注入后续 prompt。
这样它更像“历史推理经验 / claim status / evidence title memory”，而不是答案缓存。

注意：persistent memory 会跨样本积累经验，所以不能和无 memory 主实验混在一起比较。
论文里应作为独立 ablation：

```text
Supervisor Multi-Agent
vs
Supervisor Multi-Agent + Persistent Memory
```

## Agent Prompt

不同 agent 使用不同 role prompt：

```text
Vision Agent:
Act as a medical image specialist.

Reasoning Agent:
Act as a clinical reasoning expert.

Critic Agent:
Check logical errors.

Supervisor Agent:
Act as a supervisor for a multimodal medical reasoning system.

Verifier Agent:
Verify the reasoning claims against retrieved evidence.
```

## Claim Status

Phase 4 所有带 reasoning 的记录都会写入 `claim_statuses`，避免系统把 hypothesis 当成 fact：

```json
{
  "claim": "possible pneumonia",
  "status": "HYPOTHESIS"
}
```

状态含义：

```text
OBSERVED: Vision Agent 直接观察到的图像 finding。
SUPPORTED: Verifier 认为 evidence 支持的 claim。
HYPOTHESIS: Reasoning Agent 提出的候选判断，尚未完成验证。
UNSUPPORTED: 当前 evidence 不足以支持的 claim。
CONTRADICTED: 与 retrieved evidence 冲突的 claim。
```

职责划分：

```text
Vision Agent -> 主要输出 OBSERVED
Reasoning Agent -> 输出 OBSERVED / SUPPORTED / HYPOTHESIS，并明确 HYPOTHESIS 不能当作事实
Verifier Agent -> 把 claims 更新为 SUPPORTED / UNSUPPORTED / CONTRADICTED
```

## 指标

Phase 4 记录：

```text
Accuracy
Hallucination Rate
Reasoning Score
Tool Selection Accuracy
Evidence Quality Score
Mean State Compression Ratio
Mean Persistent Memory Hits
Error Attribution
Latency
Input / output tokens
```

当前 `Hallucination Rate` 和 `Reasoning Score` 是规则版启发式指标：

- `Hallucination Rate`：critic/verifier 标记 unsupported 或 contradicted 时计入。
- `Claim Status Counts`：统计 OBSERVED / SUPPORTED / HYPOTHESIS /
  UNSUPPORTED / CONTRADICTED 的数量。
- `Mean State Compression Ratio`：压缩后 agent memory 字符数 / 原始 agent
  输出字符数，越低表示共享状态越短。
- `Mean Persistent Memory Hits`：平均每个样本检索到多少条历史 memory。
- `Error Attribution`：错误样本自动归因到 Perception / Retrieval /
  Reasoning / Verification / Routing / State 六类。
- `Reasoning Score`：检查 Vision / Reasoning / Claims / Final Answer 结构是否完整。
- `Tool Selection Accuracy`：比较 Supervisor 输出的 `Selected Tools` 和
  `expected_selected_tools` 是否一致。

后续论文主结果可以再加入 GPT-based evaluator 或人工标注。

## 配置文件

Mock smoke：

```text
configs/experiments/exp04_fixed_multi_agent.yaml
configs/experiments/exp04_supervisor_multi_agent.yaml
```

AutoDL 4090D Qwen：

```text
configs/experiments/exp04_fixed_multi_agent_qwen_7b_4090d_20.yaml
configs/experiments/exp04_fixed_multi_agent_qwen_7b_4090d_full.yaml
configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_20.yaml
configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_full.yaml
configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_memory_20.yaml
configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_memory_full.yaml
```

## 运行命令

本地 mock smoke：

```bash
python scripts/run_multi_agent.py --config configs/experiments/exp04_fixed_multi_agent.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent.yaml
```

AutoDL 4090D 先跑 20 条：

```bash
python scripts/run_multi_agent.py --config configs/experiments/exp04_fixed_multi_agent_qwen_7b_4090d_20.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_20.yaml
```

20 条稳定后跑 full：

```bash
python scripts/run_multi_agent.py --config configs/experiments/exp04_fixed_multi_agent_qwen_7b_4090d_full.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_full.yaml
```

Persistent Memory ablation：

```bash
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_memory_20.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_memory_full.yaml
```

## Pass Criteria

- Fixed Multi-Agent 能写出 `agent_outputs` 和固定 `agent_route`。
- Supervisor Multi-Agent 能写出 `selected_tools`、`expected_selected_tools`、
  `expected_agent_route` 和实际 `agent_route`。
- Multi-Agent 记录必须写出 `shared_state` 和 `state_compression`。
- Persistent Memory 实验必须写出 `memory_records` 和 `memory_write_record`。
- 下游 agent prompt 必须读取 `Shared State` 中的 `compressed_context`。
- Supervisor 路径包含 Retrieval Agent 和 Verifier Agent。
- metrics 包含 `accuracy`、`hallucination_rate`、`mean_reasoning_score`、
  `mean_tool_selection_accuracy`、`mean_state_compression_ratio`、
  `mean_persistent_memory_hits`。
- Qwen 配置保持 `training_epochs: 0`。
- full test 使用 451 条 VQA-RAD test samples。

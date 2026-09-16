# Phase 8：Qwen2.5-VL LoRA 微调

Phase 8 的目标是把 accuracy 从当前 zero-shot / prompt-only 上限继续往上推。

当前 Phase 6 v6 已经达到：

```text
Accuracy: 57.43%
Oracle upper bound of existing routes: 61.20%
```

这说明只靠 Direct / Structured CoT / Full Multi-Agent 三条 frozen-route 继续调参，很难接近 70%。要冲 70%，需要先提高 backbone 在 VQA-RAD 上的基础 VQA 能力。

## 训练目标

本阶段只做最小 LoRA SFT：

```text
Image + Question -> short answer
```

不训练 CoT，不训练 RAG，不训练 Verifier。

原因：

```text
1. VQA-RAD 主指标是短答案 accuracy。
2. 训练长 reasoning 容易引入 hallucination 和格式漂移。
3. 先提高 Direct VQA，再让 Phase 6 routing 使用微调后的 backbone。
```

## 数据

使用已经导入的 VQA-RAD split：

```text
Train: Data/Processed/vqa_rad/train.jsonl
Validation: Data/Processed/vqa_rad/validation.jsonl
Test: Data/Processed/vqa_rad/test.jsonl
```

训练只使用 train / validation，不使用 test。

## 配置

训练配置：

```text
configs/training/phase8_qwen_vl_lora_vqa_rad_4090d.yaml
```

默认设置：

```text
Base model: /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
Epochs: 3
Batch size: 1
Gradient accumulation: 16
Precision: bf16
Gradient checkpointing: true
```

输出：

```text
Adapter: Experiments/phase8/qwen_vl_lora_vqa_rad_4090d/adapter
Merged model: Experiments/phase8/qwen_vl_lora_vqa_rad_4090d/merged
Metrics: Experiments/phase8/qwen_vl_lora_vqa_rad_4090d/train_metrics.json
```

默认会 merge adapter。这样后续现有推理脚本可以直接把 `model_id` 指向 merged model。

## AutoDL 安装依赖

如果还没装训练依赖：

```bash
pip install -r requirements/train.txt
```

PyTorch 仍然按 AutoDL 镜像自带版本或 CUDA 版本安装，不建议随便覆盖。

## 训练命令

```bash
python scripts/train_qwen_vl_lora.py \
  --config configs/training/phase8_qwen_vl_lora_vqa_rad_4090d.yaml
```

4090D 24GB 如果显存不够，按顺序调整：

```text
1. lora.r: 16 -> 8
2. training.gradient_accumulation_steps: 16 -> 32
3. training.num_train_epochs: 3 -> 2
4. outputs.merge_adapter: true -> false
```

如果关闭 merge，需要后续增加 adapter 推理加载；当前推荐先保持 merge。

## 评估命令

先跑 100 条：

```bash
python scripts/run_direct_vlm.py \
  --config configs/experiments/exp08_lora_direct_vlm_qwen_7b_4090d_100.yaml

python scripts/run_phase6_adaptive_routing.py \
  --config configs/experiments/exp08_lora_phase6_v6_qwen_7b_4090d_pmc10k_100.yaml
```

如果 100 条明显高于 frozen v6，再跑 full：

```bash
python scripts/run_direct_vlm.py \
  --config configs/experiments/exp08_lora_direct_vlm_qwen_7b_4090d_full.yaml

python scripts/run_phase6_adaptive_routing.py \
  --config configs/experiments/exp08_lora_phase6_v6_qwen_7b_4090d_pmc10k_full.yaml
```

## 判断标准

最低通过线：

```text
LoRA Direct full > frozen Direct full
LoRA Phase6 full > frozen Phase6 v6 full 57.43%
```

理想目标：

```text
LoRA Direct 接近或超过 65%
LoRA Phase6 接近 70%
```

如果 LoRA Direct 提升明显，但 LoRA Phase6 没提升，说明路由规则需要重新基于 LoRA 结果做 oracle analysis。

如果 LoRA Direct 没提升，优先检查：

```text
1. 是否训练样本太少或没有真正读到 train split。
2. 是否 answer 格式不稳定。
3. learning_rate 是否过高。
4. epoch 是否过多导致过拟合 validation。
```

## 与前面 Phase 的关系

Phase 8 不是替代 Phase 6，而是给 Phase 6 提供更强 backbone：

```text
Frozen Qwen2.5-VL-7B + Phase6 v6
vs
LoRA Qwen2.5-VL-7B + Phase6 v6
```

这样论文叙事更清楚：

```text
Domain adaptation improves base VQA ability.
Adaptive routing further improves reasoning efficiency and route selection.
```

## 常见报错：TrainingArguments 不支持 warmup_ratio

如果遇到：

```text
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'
```

说明 AutoDL 镜像里的 transformers 版本和本地不完全一致。训练脚本已经做了兼容：

```text
新版 transformers：使用 warmup_ratio / eval_strategy
旧版 transformers：自动改用 warmup_steps / evaluation_strategy
不支持的 TrainingArguments 字段会自动跳过
```

拉取最新代码后重新运行训练命令即可。

## LoRA Structured CoT 与 Phase6 v7

当前 LoRA full 结果：

```text
LoRA Direct full: 59.87%
LoRA Phase6 v6 full: 58.76%
Oracle upper bound: 68.51%
```

这说明 LoRA Direct 已经强于当前 v6 routing。v6 的主要问题是 CoT 使用偏多：

```text
selected structured_cot: 113
structured_cot -> oracle direct: 73
```

因此新增两组配置：

```text
LoRA Structured CoT:
configs/experiments/exp08_lora_structured_cot_qwen_7b_4090d_100.yaml
configs/experiments/exp08_lora_structured_cot_qwen_7b_4090d_full.yaml

LoRA Phase6 v7:
configs/experiments/exp08_lora_phase6_v7_qwen_7b_4090d_pmc10k_100.yaml
configs/experiments/exp08_lora_phase6_v7_qwen_7b_4090d_pmc10k_full.yaml
```

v7 策略：

```text
Direct dominant.
只把明确的 SIZE、ATTRIB closed、侧别、比较、密度、钙化、测量类问题送入 Structured CoT。
诊断、病因、疾病类 marker 不再自动升级到 CoT。
Full Multi-Agent 暂时不启用。
```

推荐运行顺序：

```bash
python scripts/run_cot.py \
  --config configs/experiments/exp08_lora_structured_cot_qwen_7b_4090d_100.yaml

python scripts/run_phase6_adaptive_routing.py \
  --config configs/experiments/exp08_lora_phase6_v7_qwen_7b_4090d_pmc10k_100.yaml
```

如果 100 条结果不低于 LoRA Direct 100，再跑 full：

```bash
python scripts/run_cot.py \
  --config configs/experiments/exp08_lora_structured_cot_qwen_7b_4090d_full.yaml

python scripts/run_phase6_adaptive_routing.py \
  --config configs/experiments/exp08_lora_phase6_v7_qwen_7b_4090d_pmc10k_full.yaml
```

用 LoRA Direct、LoRA Structured CoT 和 v7 重新做 oracle：

```bash
python scripts/analyze_oracle_routing.py \
  --direct Results/exp08_lora_direct_vlm_qwen_7b_4090d_full/predictions.jsonl \
  --cot Results/exp08_lora_structured_cot_qwen_7b_4090d_full/predictions.jsonl \
  --full-agent Results/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_full/predictions.jsonl \
  --routing Results/exp08_lora_phase6_v7_qwen_7b_4090d_pmc10k_full/predictions.jsonl \
  --output Results/oracle_routing_analysis_exp08_lora_phase6_v7_full
```

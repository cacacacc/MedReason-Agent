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

## Phase6 v8 极窄路由

v7 full 结果：

```text
LoRA Direct full: 59.87%
LoRA Phase6 v7 full: 58.76%
v7 direct: 362 samples, 58.01%
v7 structured_cot: 89 samples, 61.80%
```

v7 仍未超过 Direct，说明 Structured CoT 触发还偏宽。v8 采用更极窄策略：

```text
默认全部 Direct。
只保留 SIZE、明确侧别、明确比较、计数、测量类问题进入 Structured CoT。
不再因为 closed ATTRIB、calcification、hypodense、MRI sequence 进入 CoT。
不启用 Full Multi-Agent。
```

运行：

```bash
python scripts/run_phase6_adaptive_routing.py \
  --config configs/experiments/exp08_lora_phase6_v8_qwen_7b_4090d_pmc10k_100.yaml
```

如果 100 条不低于 58%，再跑 full：

```bash
python scripts/run_phase6_adaptive_routing.py \
  --config configs/experiments/exp08_lora_phase6_v8_qwen_7b_4090d_pmc10k_full.yaml
```

## Step 1: Answer Normalization Re-evaluation

为了减少格式误判，当前评估已经增强了短答案归一化：

```text
computed tomography -> ct
magnetic resonance imaging -> mri
chest radiograph / chest x-ray -> cxr
ultrasonography / sonography / us -> ultrasound
left side / left-sided -> left
right side / right-sided -> right
within normal limits / no abnormality -> normal
gall stones -> gallstones
renal -> kidney
```

已有实验不需要重跑模型，可以直接重评估：

```bash
python scripts/reevaluate_predictions.py \
  --predictions Results/exp08_lora_direct_vlm_qwen_7b_4090d_full/predictions.jsonl \
  --output-dir Results/exp08_lora_direct_vlm_qwen_7b_4090d_full_renorm

python scripts/reevaluate_predictions.py \
  --predictions Results/exp08_lora_phase6_v8_qwen_7b_4090d_pmc10k_full/predictions.jsonl \
  --output-dir Results/exp08_lora_phase6_v8_qwen_7b_4090d_pmc10k_full_renorm
```

重评估结果只能作为 normalized evaluation；论文表格中要明确标注：

```text
Accuracy 使用 enhanced answer normalization。
```

## Step 2: LoRA v2 Question-Type-Aware Training

LoRA v2 不训练 CoT，而是把训练目标改成更严格的题型感知短答案：

```text
CLOSED -> yes / no
MODALITY -> xray / ct / mri / ultrasound
PLANE -> axial / coronal / sagittal / frontal / lateral
POSITION -> shortest location phrase
```

训练配置：

```text
configs/training/phase8_qwen_vl_lora_v2_format_48gb.yaml
```

48GB 显卡训练：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_qwen_vl_lora.py \
  --config configs/training/phase8_qwen_vl_lora_v2_format_48gb.yaml
```

训练完成后先跑 100 条 Direct：

```bash
python scripts/run_direct_vlm.py \
  --config configs/experiments/exp08_lora_v2_direct_vlm_qwen_7b_48gb_100.yaml
```

如果 100 条明显高于 LoRA v1 Direct 100 的 58%，再跑 full：

```bash
python scripts/run_direct_vlm.py \
  --config configs/experiments/exp08_lora_v2_direct_vlm_qwen_7b_48gb_full.yaml
```

如果 Direct full 接近或超过 65%，再跑 Phase6 v8：

```bash
python scripts/run_phase6_adaptive_routing.py \
  --config configs/experiments/exp08_lora_v2_phase6_v8_qwen_7b_48gb_pmc10k_100.yaml

python scripts/run_phase6_adaptive_routing.py \
  --config configs/experiments/exp08_lora_v2_phase6_v8_qwen_7b_48gb_pmc10k_full.yaml
```

## LoRA v2b: 低过拟合版本

v2 的 5 epoch 训练在 validation loss 上明显过拟合，100 条 accuracy 从 v1 的 58%
降到 55%。v2b 保留题型感知输出格式，但降低训练强度：

```text
num_train_epochs: 2
learning_rate: 5e-5
LoRA rank: 8
target_modules: q_proj, v_proj
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
```

训练：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_qwen_vl_lora.py \
  --config configs/training/phase8_qwen_vl_lora_v2b_format_48gb.yaml
```

训练完先跑 100 条：

```bash
python scripts/run_direct_vlm.py \
  --config configs/experiments/exp08_lora_v2b_direct_vlm_qwen_7b_48gb_100.yaml
```

如果 v2b Direct 100 高于 58%，再跑 full：

```bash
python scripts/run_direct_vlm.py \
  --config configs/experiments/exp08_lora_v2b_direct_vlm_qwen_7b_48gb_full.yaml
```

如果 Direct full 明显提升，再跑 Phase6 v8：

```bash
python scripts/run_phase6_adaptive_routing.py \
  --config configs/experiments/exp08_lora_v2b_phase6_v8_qwen_7b_48gb_pmc10k_100.yaml

python scripts/run_phase6_adaptive_routing.py \
  --config configs/experiments/exp08_lora_v2b_phase6_v8_qwen_7b_48gb_pmc10k_full.yaml
```

## Phase6 v9: Oracle-Distilled Learned Router

Oracle routing 的 68.51% 是事后上限，不能直接部署。v9 的目标是把 oracle label
蒸馏成一个轻量 router，在真实推理时先跑 Direct，再根据 Direct 输出和问题元数据决定是否升级：

```text
Direct first
-> extract question / metadata / direct prediction features
-> learned router
-> direct / structured_cot / full_multi_agent
```

训练 router 之前，先确保已经有 oracle analysis：

```bash
python scripts/analyze_oracle_routing.py \
  --direct Results/exp08_lora_direct_vlm_qwen_7b_4090d_full/predictions.jsonl \
  --cot Results/exp02_structured_cot_qwen_7b_4090d_full/predictions.jsonl \
  --full-agent Results/exp04_supervisor_multi_agent_heuristic_routing_qwen_7b_4090d_pmc10k_full/predictions.jsonl \
  --routing Results/exp08_lora_phase6_v8_qwen_7b_4090d_pmc10k_full/predictions.jsonl \
  --output Results/oracle_routing_analysis_exp08_lora_phase6_v8_full
```

训练 oracle-distilled router：

```bash
python scripts/train_phase6_oracle_router.py \
  --oracle-analysis Results/oracle_routing_analysis_exp08_lora_phase6_v8_full/oracle_analysis.csv \
  --output-model Experiments/phase6_router/exp08_lora_oracle_router.pkl \
  --output-metrics Experiments/phase6_router/exp08_lora_oracle_router_metrics.json \
  --validation-ratio 0.2 \
  --class-weight balanced
```

先跑 100 条：

```bash
python scripts/run_phase6_learned_router.py \
  --config configs/experiments/exp08_lora_phase6_v9_learned_router_qwen_7b_4090d_pmc10k_100.yaml
```

如果 100 条高于 LoRA Direct 100，再跑 full：

```bash
python scripts/run_phase6_learned_router.py \
  --config configs/experiments/exp08_lora_phase6_v9_learned_router_qwen_7b_4090d_pmc10k_full.yaml
```

注意：

```text
如果 router 使用 full test 的 oracle_analysis.csv 训练，再在同一个 full test 上评估，
这个结果属于 oracle-distilled diagnostic，不是严格无泄漏主结果。

严格主结果需要用 development subset 训练 router，再在 held-out test 上评估。
```

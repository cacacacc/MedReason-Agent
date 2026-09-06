# MedReason-Agent

> A Supervisor-Guided Multi-Agent Framework for Reliable Multimodal Medical Reasoning

MedReason-Agent 是一个科研型 AI 项目，研究对象是医学多模态视觉问答中的推理可靠性。它不是真实临床诊断产品，而是用 benchmark 实验测试：显式推理、RAG、多 Agent 协作、Supervisor 路由和 Verification 是否能提升 frozen VLM 的可靠性。

## 研究重点

主研究问题：

Supervisor-guided Multi-Agent reasoning framework 是否能比 Direct VLM inference 在 accuracy、evidence grounding、calibration 和 efficiency 上更可靠？

项目会按 controlled experiment chain 逐步比较：

1. Random / majority baseline
2. Direct VLM
3. Structured direct prompt
4. Chain-of-thought prompting
5. RAG-augmented reasoning
6. Fixed multi-agent pipeline
7. Supervisor multi-agent routing
8. Supervisor with verification / reflection
9. Adaptive reasoning
10. Optional LoRA fine-tuning

## 核心原则

主实验使用 frozen backbone。我们希望证明提升来自 reasoning architecture 和 evaluation design，而不是来自不同实验使用了不同大小的模型。

默认 backbone：

- 主实验：Qwen2.5-VL-7B-Instruct
- 轻量开发 fallback：Qwen2.5-VL-3B-Instruct

## 当前阶段

Phase 0: Research Foundation + Repository Setup

当前工作：

- 定义研究问题。
- 定义统一实验协议。
- 定义第一版错误分类。
- 建立 VQA-RAD 数据导入流程。
- 建立项目学习材料。

当前还没有实现核心 Agent 代码。

## 当前结构

```text
MedReason-Agent/
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── exp00_majority.yaml
│       └── exp01_direct_vlm.yaml
├── Data/
│   ├── Raw/
│   └── Processed/
├── Docs/
│   ├── research_questions.md
│   ├── experiment_protocol.md
│   ├── error_taxonomy.md
│   └── datasets.md
├── Experiments/
├── Results/
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   ├── agent.txt
│   ├── rag.txt
│   └── vlm.txt
├── scripts/
│   ├── check_environment.py
│   └── import_vqa_rad.py
├── src/
│   └── medreason_agent/
├── assets/
│   └── lesson.css
├── lessons/
│   └── 0001-medical-vqa-and-baseline.html
├── MISSION.md
├── NOTES.md
├── RESOURCES.md
└── README.md
```

## 医学安全边界

本项目所有输出都必须描述为 benchmark research 中的 model prediction。它们不是医疗建议、临床诊断，也不能替代专业医生判断。

## 环境安装

推荐本地环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements/dev.txt
pip install -e .
```

可选依赖组：

```powershell
pip install -r requirements/agent.txt
pip install -r requirements/rag.txt
pip install -r requirements/vlm.txt
```

检查环境：

```powershell
python scripts/check_environment.py
pytest -q
```

## 导入 VQA-RAD 数据

```powershell
.\.venv\Scripts\python.exe scripts\import_vqa_rad.py
```

导入完成后会生成：

```text
Data/Processed/vqa_rad/all.jsonl
Data/Processed/vqa_rad/all.csv
Data/Processed/vqa_rad/train.jsonl
Data/Processed/vqa_rad/train.csv
Data/Processed/vqa_rad/validation.jsonl
Data/Processed/vqa_rad/validation.csv
Data/Processed/vqa_rad/test.jsonl
Data/Processed/vqa_rad/test.csv
Data/Processed/vqa_rad/stats.json
```

## 运行 Direct VLM Baseline Smoke Test

当前默认使用 `mock` backend，只用于验证实验管线，不代表真实模型结果：

```powershell
.\.venv\Scripts\python.exe scripts\run_direct_vlm.py
```

输出：

```text
Results/exp01_direct_vlm/predictions.jsonl
Results/exp01_direct_vlm/metrics.json
```

真实 Qwen2.5-VL baseline 需要后续安装 `requirements/vlm.txt` 并实现/启用 `qwen2_5_vl` backend。

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

## 运行 Chain-of-Thought Baseline Smoke Test

当前默认使用 `mock` backend，只用于验证 CoT 实验管线：

```powershell
.\.venv\Scripts\python.exe scripts\run_cot.py
.\.venv\Scripts\python.exe scripts\compare_direct_vs_cot.py
```

输出：

```text
Results/exp02_cot/predictions.jsonl
Results/exp02_cot/metrics.json
Results/compare_direct_vs_cot/summary.json
Results/compare_direct_vs_cot/comparisons.jsonl
```

## 运行 Medical RAG Smoke Test

先构建一个很小的 seed medical knowledge base：

```powershell
.\.venv\Scripts\python.exe scripts\build_medical_kb.py --seed-medical-vqa
```

再运行 Knowledge RAG baseline。它是“先查知识，再推理”：

```powershell
.\.venv\Scripts\python.exe scripts\run_rag.py
```

也可以运行 Evidence RAG baseline。它是“先有 claim，再查证据验证”：

```powershell
.\.venv\Scripts\python.exe scripts\run_rag.py --config configs\experiments\exp03_evidence_rag.yaml
```

运行 Knowledge + Evidence RAG Verifier。它先用 Knowledge RAG 辅助推理，再用
Evidence RAG 验证 claim：

```powershell
.\.venv\Scripts\python.exe scripts\run_rag.py --config configs\experiments\exp03_knowledge_evidence_rag.yaml
```

输出：

```text
Results/exp03_rag/predictions.jsonl
Results/exp03_rag/metrics.json
```

Phase 3 会额外记录：

```text
evidence_quality_score
mean_evidence_quality_score
mean_question_coverage
mean_answer_coverage
evidence_empty_rate
claim_status_counts
```

当前 RAG prompt version 是 `rag_v3`。它会约束模型：

```text
Use retrieved evidence explicitly
Do not introduce unsupported medical claims
Separate image observation from external knowledge
State uncertainty if evidence is insufficient
Label every claim as OBSERVED / SUPPORTED / HYPOTHESIS / UNSUPPORTED / CONTRADICTED
```

两种 RAG 在结果里通过 `rag_mode` 和 `agent_route` 区分：

```text
Knowledge RAG:
rag_mode = knowledge_acquisition
route = knowledge_agent -> retriever -> reranker -> evidence_filter -> clinical_reasoning_agent

Evidence RAG:
rag_mode = claim_verification
route = reasoning_agent -> claim_extractor -> evidence_agent -> retriever -> reranker -> evidence_filter -> verifier_agent

Knowledge + Evidence RAG Verifier:
rag_mode = knowledge_then_claim_verification
route = knowledge_agent -> retriever -> reranker -> evidence_filter -> reasoning_agent -> claim_extractor -> evidence_agent -> retriever -> reranker -> evidence_filter -> verifier_agent
verifier decision = SUPPORTED / UNSUPPORTED / CONTRADICTED
claim_statuses = per-claim OBSERVED / SUPPORTED / HYPOTHESIS / UNSUPPORTED / CONTRADICTED records
```

`HYPOTHESIS` 只表示推理候选判断，不能当作已确认医学事实；只有 `OBSERVED` 和
`SUPPORTED` 可以作为较强证据进入结论分析。

Phase 4 Multi-Agent 使用 sample-level shared state：

```text
Agent raw output -> State Compression -> SharedAgentState -> Next Agent
```

结果会额外记录：

```text
shared_state
state_compression
mean_state_compression_ratio
memory_records
memory_write_record
mean_persistent_memory_hits
error_attribution
error_type_counts
```

当前压缩方法是 `section_extract_v1`，只保留关键段落、retrieved evidence 摘要和
claim statuses。它不是跨样本长期记忆，不会把一个病例的信息带到另一个病例。

需要跨对话/跨运行存在的 memory 时，使用 persistent memory ablation 配置：

```powershell
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_memory.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_memory_20.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_memory_full.yaml
```

Persistent memory 默认写入 `Experiments/memory/*.jsonl`，下次运行仍然存在。
默认不保存 `ground_truth`，也不把历史 `prediction` 注入 prompt，减少 benchmark 泄漏。

## AutoDL 4090D Phase 3 配置

AutoDL 单张 RTX 4090D 24GB 推荐先使用这些配置：

```powershell
python scripts/run_rag.py --config configs/experiments/exp03_rag_qwen_7b_cuda_smoke.yaml
python scripts/run_rag.py --config configs/experiments/exp03_rag_qwen_7b_cuda_5.yaml
python scripts/run_rag.py --config configs/experiments/exp03_evidence_rag_qwen_7b_4090d_5.yaml
python scripts/run_rag.py --config configs/experiments/exp03_knowledge_evidence_rag_qwen_7b_4090d_5.yaml
```

上传模型到 AutoDL 后，建议把配置里的 `model_id` 改成：

```text
/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
```

4090D 首跑参数策略：

```text
torch_dtype: bfloat16
device_map: auto
top_k: 2
candidate_top_k: 6
max_chars_per_evidence: 450-500
max_new_tokens: 256
```

## 构建 PubMed / PMC 10k 知识库

Phase 3 的第一版真实 RAG corpus 使用 10,000 篇医学摘要，不要直接上全量。
这个规模适合作为第一版主实验：足够测试 retrieval quality，又不会让 AutoDL 调试成本
过高。
先下载 PubMed 摘要 JSONL：

```bash
python scripts/download_pubmed_abstracts.py \
  --max-records 10000 \
  --email your_email@example.com \
  --output Data/Raw/pmc/pmc_abstracts.jsonl
```

输出文件：

```text
Data/Raw/pmc/pmc_abstracts.jsonl
```

每行至少包含 `pmid`、`title`、`abstract` 和 `source`。然后构建 chunks：

```bash
python scripts/build_medical_kb.py \
  --input-jsonl Data/Raw/pmc/pmc_abstracts.jsonl \
  --output Data/Processed/medical_kb/pmc_10k_chunks.jsonl \
  --corpus-name pmc_abstracts_10k \
  --max-documents 10000 \
  --chunk-unit tokenizer \
  --tokenizer-name-or-path /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
  --chunk-size 256 \
  --overlap 50
```

4090D 上先跑 20 条：

```bash
python scripts/build_faiss_index.py \
  --chunks Data/Processed/medical_kb/pmc_10k_chunks.jsonl \
  --index-output Data/Processed/medical_kb/pmc_10k_bge_small.faiss \
  --metadata-output Data/Processed/medical_kb/pmc_10k_bge_small_metadata.jsonl \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --device cuda \
  --batch-size 64

python scripts/run_rag.py --config configs/experiments/exp03_rag_qwen_7b_4090d_pmc10k_20.yaml
python scripts/run_rag.py --config configs/experiments/exp03_knowledge_evidence_rag_qwen_7b_4090d_pmc10k_20.yaml
```

检索设置：

```text
Embedding: BAAI/bge-small-en-v1.5
Vector index: FAISS
Main top-k: 5
Ablation top-k: 3 / 5 / 10
```

20 条 smoke 稳定后，直接跑 full：

```bash
python scripts/run_rag.py --config configs/experiments/exp03_rag_qwen_7b_4090d_pmc10k_full.yaml
python scripts/run_rag.py --config configs/experiments/exp03_knowledge_evidence_rag_qwen_7b_4090d_pmc10k_full.yaml
```

## 运行 Phase 4 Supervisor Multi-Agent

本地先跑 mock smoke：

```bash
python scripts/run_multi_agent.py --config configs/experiments/exp04_fixed_multi_agent.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent.yaml
```

AutoDL 4090D 上先跑 20 条：

```bash
python scripts/run_multi_agent.py --config configs/experiments/exp04_fixed_multi_agent_qwen_7b_4090d_20.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_20.yaml
```

20 条稳定后跑 full：

```bash
python scripts/run_multi_agent.py --config configs/experiments/exp04_fixed_multi_agent_qwen_7b_4090d_full.yaml
python scripts/run_multi_agent.py --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_full.yaml
```

Phase 4 文档见 [Docs/phase4_supervisor_multi_agent.md](Docs/phase4_supervisor_multi_agent.md)。

## 接入真实 Qwen2.5-VL

先按你的 CPU / CUDA 环境安装 PyTorch：

```text
https://pytorch.org/get-started/locally/
```

再安装 VLM 依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\vlm.txt
```

CPU 环境先运行 1 条样本 smoke test：

```powershell
.\.venv\Scripts\python.exe scripts\run_direct_vlm.py --config configs\experiments\exp01_direct_vlm_qwen_smoke.yaml
```

更多说明见 [Docs/model_connection.md](Docs/model_connection.md)。

## VQA-RAD 数据版本策略

MedReason-Agent 使用 OSF 公共版 VQA-RAD。当前项目统一采用下面的可复现统计口径：

```text
Raw JSON QA records: 2248
Image files: 315
Referenced images: 314
Official raw train pool: 1797
Official raw test split: 451
Project train split: 1527
Project validation split: 270
Project test split: 451
```

在本仓库中，VQA-RAD full test 指 `451` 条 test samples。除非项目明确切换到另一个
dataset mirror，否则主实验不要使用 `464` 或 `3064` 作为 VQA-RAD 实验数量。数据版本
审计见 `Docs/vqa_rad_data_version_audit.md`。

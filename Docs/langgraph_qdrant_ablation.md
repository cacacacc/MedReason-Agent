# LangGraph and Qdrant Controlled Ablation

This extension adds two optional Phase 3 variables:

- `method.orchestrator`: `python` or `langgraph`
- `retrieval.retriever`: `faiss` or `qdrant`

The control principle is:

- To evaluate LangGraph, keep retriever fixed as FAISS.
- To evaluate Qdrant, keep orchestrator fixed as Python.
- Keep dataset split, sample count, model, prompt, embedding model, `top_k`, and
  `candidate_top_k` unchanged.

## Build Qdrant Collection

```bash
python scripts/build_qdrant_collection.py \
  --chunks Data/Processed/medical_kb/pmc_10k_chunks.jsonl \
  --qdrant-path Data/Processed/medical_kb/qdrant_pmc_10k_bge_small \
  --collection-name pmc_10k_bge_small \
  --embedding-model /root/autodl-tmp/models/bge-small-en-v1.5 \
  --device cuda \
  --force-recreate
```

## Run Controlled 20-Sample Smoke Ablations

Baseline: Python orchestration + FAISS.

```bash
python scripts/run_rag.py \
  --config configs/experiments/exp03_rag_qwen_7b_4090d_pmc10k_20.yaml
```

LangGraph only: LangGraph orchestration + FAISS.

```bash
python scripts/run_rag.py \
  --config configs/experiments/exp03_rag_langgraph_qwen_7b_4090d_pmc10k_20.yaml
```

Qdrant only: Python orchestration + Qdrant.

```bash
python scripts/run_rag.py \
  --config configs/experiments/exp03_rag_qdrant_qwen_7b_4090d_pmc10k_20.yaml
```

Combined system: LangGraph orchestration + Qdrant.

```bash
python scripts/run_rag.py \
  --config configs/experiments/exp03_rag_langgraph_qdrant_qwen_7b_4090d_pmc10k_20.yaml
```

## Compare Metrics

```bash
python scripts/compare_rag_ablation.py \
  --metrics \
  Results/exp03_rag_qwen_7b_4090d_pmc10k_20/metrics.json \
  Results/exp03_rag_langgraph_qwen_7b_4090d_pmc10k_20/metrics.json \
  Results/exp03_rag_qdrant_qwen_7b_4090d_pmc10k_20/metrics.json \
  Results/exp03_rag_langgraph_qdrant_qwen_7b_4090d_pmc10k_20/metrics.json \
  --output Results/phase3_langgraph_qdrant_ablation.md
```

## Interpretation

LangGraph should not be expected to improve answer accuracy by itself, because it
keeps the same prompts, retrieval evidence, and VLM calls. Its value is clearer
state transitions, auditable routes, and safer future branching or retry logic.

Qdrant can change retrieval ranking only if the vector search behavior differs
from the FAISS index under the same normalized BGE embeddings. On the 10k corpus,
the expected advantage is mainly operational extensibility rather than guaranteed
accuracy improvement.

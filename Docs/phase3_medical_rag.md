# Phase 3: Medical RAG

## Phase Goal

Phase 3 studies whether retrieved medical evidence can improve multimodal
medical VQA reliability compared with Direct VLM and CoT baselines.

The target architecture is:

```text
Question -> Retriever -> Reranker -> Evidence Filter -> VLM Reasoning -> Answer
```

## Why

Direct VLM and CoT depend mainly on the model's internal knowledge and visual
understanding. Medical RAG adds external evidence before generation. This may
reduce hallucination, but it may also hurt performance if retrieval returns
irrelevant evidence.

## Theory

`RAG`

RAG means Retrieval-Augmented Generation. In simple terms, the model first looks
up relevant knowledge, then answers with that knowledge as context.

Project example: for a question about an aortic aneurysm, the retriever may
return evidence about widened aortic contour or mural calcification. The VLM
then answers using both the image and this evidence.

Why the project needs it: medical questions often require knowledge that may not
be fully visible in the image. RAG tests whether external medical evidence helps
the frozen VLM reason more reliably.

`Retriever`

A retriever is the component that searches the knowledge base and returns the
most relevant evidence.

Current implementation: `KeywordRetriever`, a lightweight keyword retriever used
to validate the RAG experiment pipeline.

Target implementation: BGE embedding + Qdrant vector retrieval.

`Retrieved Evidence`

Retrieved evidence is the text passed into the VLM prompt. It must be saved in
each prediction record so we can later check whether the model used relevant or
irrelevant evidence.

`Reranker`

The reranker re-scores candidate chunks returned by the retriever. The current
implementation uses a lightweight keyword-overlap reranker so that RTX 4070 can
run the pipeline without loading another large model.

`Evidence Filter`

The evidence filter keeps only high-quality reranked evidence before the prompt
is built. This reduces prompt length and avoids sending obviously irrelevant
chunks into the reasoning agent.

## Files

```text
CREATE
Docs/phase3_medical_rag.md
src/medreason_agent/retrieval/__init__.py
src/medreason_agent/retrieval/chunking.py
src/medreason_agent/retrieval/keyword.py
src/medreason_agent/retrieval/rerank.py
src/medreason_agent/prompts/rag.py
scripts/build_medical_kb.py
scripts/run_rag.py
configs/experiments/exp03_rag.yaml
configs/experiments/exp03_rag_qwen_7b_cuda_smoke.yaml
configs/experiments/exp03_rag_qwen_7b_cuda_5.yaml
configs/experiments/exp03_rag_qwen_7b_cuda_20.yaml
configs/experiments/exp03_rag_qwen_7b_cuda_100.yaml
configs/experiments/exp03_rag_qwen_7b_cuda_full.yaml
tests/test_rag_prompt.py
tests/test_retrieval.py

MODIFY
scripts/run_rag.py
configs/experiments/exp03_rag*.yaml

DELETE
None
```

## Current Implementation Scope

The current implementation is a runnable Phase 3 scaffold:

- Builds a small seed medical knowledge base for smoke testing.
- Splits documents into chunks.
- Retrieves candidate evidence with a deterministic keyword retriever.
- Reranks candidate evidence with a lightweight keyword reranker.
- Filters reranked evidence before sending it to the reasoning agent.
- Builds a RAG prompt with the filtered high-quality evidence.
- Saves `retrieved_evidence`, `tool_calls`, `reasoning_output`, and answer
  metrics.

This is not yet the final PubMed/BGE/Qdrant RAG system.

## Next Implementation Step

Replace or extend the keyword retriever with:

```text
PubMed abstracts -> chunk -> BGE embedding -> Qdrant -> candidate evidence
-> reranker -> evidence filter -> final evidence
```

The first real corpus target should be smaller than 10,000 abstracts for smoke
testing, then gradually scale:

```text
Seed KB: pipeline validation only
PubMed smoke: 100 abstracts
PubMed dev: 1,000 abstracts
PubMed first main KB: 10,000 abstracts
```

## Dataset Scale

VQA-RAD experiment scale follows the project-wide policy:

```text
Smoke: 1-20 test samples
Dev: 100 test samples
Full: 451 test samples
```

## Pass Criteria

- `scripts/build_medical_kb.py --seed-medical-vqa` creates a local chunk file.
- `scripts/run_rag.py` writes predictions and metrics.
- Each prediction record contains non-empty `retrieved_evidence` when retrieval
  finds evidence.
- `reasoning_output` preserves the full RAG reasoning text.
- `prediction` is extracted from `Final Answer:`.
- Tests pass.

## Research Questions

1. Why can RAG reduce hallucination?
2. Why can irrelevant retrieved evidence make answers worse?
3. Why must we save retrieved evidence for every prediction?
4. Why should keyword retrieval not be treated as the final RAG method?

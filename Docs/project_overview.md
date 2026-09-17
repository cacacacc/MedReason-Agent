# Project Overview

## One-Sentence Purpose

MedReason-Agent investigates how to make multimodal medical VQA more reliable by
controlling reasoning depth, retrieval, verification, and lightweight adaptation
around Qwen2.5-VL on VQA-RAD.

## What the Project Is

This is a research-style benchmark project, not a clinical product. It evaluates
different reasoning workflows for medical visual question answering:

- direct VLM answering;
- Chain-of-Thought and structured prompting;
- RAG with external medical abstracts;
- fixed and supervisor multi-agent pipelines;
- verifier and self-reflection loops;
- adaptive routing across reasoning depths;
- LoRA adaptation for short-answer medical VQA;
- new optional LangGraph and Qdrant ablation paths.

The common thread is controlled comparison. The project tries to keep the
dataset, model backbone, prompt contracts, and metrics explicit so that each
phase answers a specific research question rather than just adding components.

## What Problem It Addresses

Medical VQA systems can fail in several ways:

- visual perception errors;
- unsupported medical claims;
- verbose answers that do not match short benchmark labels;
- reasoning drift introduced by long CoT;
- irrelevant retrieved evidence;
- verifier decisions that do not translate into better final answers;
- unnecessary agent calls that increase cost and error propagation.

The project asks whether a system can decide when to use simple direct answering
and when to use deeper reasoning.

## Current Research Thesis

The current evidence supports this thesis:

```text
For VQA-RAD with Qwen2.5-VL, reliability improves more from adaptive reasoning
control than from always adding RAG, multi-agent reasoning, or verifier loops.
```

In plain language:

```text
The supervisor should not always make the system more complicated. Its most
useful role is deciding when not to use extra reasoning.
```

## Pipeline Map

```text
Image + Question
  -> Direct VLM baseline
  -> Structured CoT candidate
  -> RAG / evidence retrieval candidate
  -> Multi-agent / verifier candidate
  -> Adaptive router chooses reasoning depth
  -> Final answer + metrics + audit records
```

The project records both answer quality and process quality:

- final prediction;
- reasoning output;
- retrieved evidence;
- claim statuses;
- agent route;
- tool calls;
- latency;
- token counts;
- error attribution;
- route-level metrics.

## Dataset and Model

Main dataset:

```text
VQA-RAD OSF public release
Project test split: 451 samples
```

Main backbone:

```text
Qwen2.5-VL-7B-Instruct
```

Retrieval corpus:

```text
PMC / PubMed 10k medical abstracts
```

Main dense retrieval setup so far:

```text
BGE-small embedding + FAISS
```

New optional retrieval ablation:

```text
BGE-small embedding + Qdrant
```

## What Has Been Answered

### 1. Direct VLM is a strong baseline.

Direct Qwen2.5-VL achieved roughly 49-50% accuracy on the 451-sample VQA-RAD
test split. It is especially strong on closed questions and weaker on open
descriptive questions.

### 2. Naive CoT does not improve the full test.

Vanilla and structured CoT underperformed direct answering on the full split.
This suggests that asking the model to reason more can introduce drift.

### 3. The first RAG design did not improve final-answer accuracy.

Knowledge RAG and Knowledge + Evidence RAG created useful evidence records, but
both underperformed direct answering. The main likely issue is that retrieved
abstracts are often generic and not image-specific enough.

### 4. Fixed multi-agent and broad supervisor pipelines did not beat Direct VLM.

Adding more agents improved auditability but not accuracy. Multi-agent pipelines
can amplify intermediate mistakes and increase latency.

### 5. Verification is useful for audit, but not yet for answer correction.

Self-reflection and separate verifier paths recorded process signals, but they
caused answer regressions and did not improve final accuracy.

### 6. Adaptive routing is the strongest frozen-backbone result.

Phase 6 found that a direct/structured-CoT dominant routing policy performs best
among frozen-backbone methods.

Key tracked result:

```text
Phase 6 v6 full accuracy: 57.43%
Mean agent calls: 1.00
```

This supports the central conclusion that choosing reasoning depth matters more
than always using a full multi-agent pipeline.

### 7. LoRA improves the base VQA ability.

Phase 8 LoRA direct inference improved over the frozen direct baseline. This
suggests that domain adaptation and routing are complementary:

```text
LoRA improves the base answerer.
Adaptive routing decides when extra reasoning is useful.
```

## What Is Not Answered Yet

### LangGraph

LangGraph has now been implemented as an optional RAG orchestrator, but no
completed accuracy comparison has been recorded yet. It should be presented as
an implemented ablation path, not as a proven improvement.

### Qdrant

Qdrant has now been implemented as an optional vector retriever, but no completed
FAISS-vs-Qdrant result has been recorded yet. It should be presented as a
retrieval infrastructure ablation, not as a proven accuracy gain.

### Learned Routing

The project includes direction for oracle-distilled learned routing, but strict
held-out evaluation is still needed before claiming it as an unbiased main
result.

## What the Project Contributes

The project contributes an experimental story:

```text
1. Establish direct VLM baseline.
2. Show naive CoT is not enough.
3. Show generic RAG and broad multi-agent pipelines can hurt.
4. Use those negative results to motivate adaptive routing.
5. Show adaptive routing improves frozen-backbone accuracy and efficiency.
6. Explore LoRA as a complementary way to raise the base model ceiling.
7. Add LangGraph and Qdrant as controlled infrastructure ablations.
```

This is a defensible research narrative because it includes negative findings,
ablation logic, and a clear reason why the final design changed over time.

## Resume-Safe Technical Stack

Strong, already supported:

```text
Qwen2.5-VL
Medical VQA
VQA-RAD
RAG
BGE-small embeddings
FAISS
Multi-agent reasoning
Verifier / self-reflection experiments
Adaptive routing
LoRA fine-tuning
```

Implemented but should be described carefully:

```text
LangGraph orchestration ablation
Qdrant retrieval ablation
```

Avoid overstating:

```text
Do not claim Qdrant improved results until the ablation has metrics.
Do not claim LangGraph improved accuracy until the ablation has metrics.
Do not claim multi-agent reasoning universally improves medical VQA.
```

## Recommended Interview Answer

```text
The goal of my project was to study reliable medical VQA with a controlled
Qwen2.5-VL pipeline. I compared direct answering, CoT, RAG, multi-agent
reasoning, verification, adaptive routing, and LoRA. The main research question
was whether reliability comes from adding more reasoning modules or from choosing
the right reasoning depth per sample. The experiments showed that naive CoT,
generic RAG, and broad multi-agent workflows often hurt accuracy. The strongest
frozen-backbone result came from adaptive routing, which mostly used direct
answering and selectively used structured CoT. So the key conclusion is that
reasoning control is more important than always making the pipeline more complex.
```

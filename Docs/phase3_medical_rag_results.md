# Phase 3 Medical RAG Results

## Experiment

```text
Backbone: Qwen/Qwen2.5-VL-7B-Instruct
Local model path: /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
Backend: qwen2_5_vl
Training epochs: 0
Dataset: VQA-RAD OSF public release
Split: test
Full test samples: 451
Medical corpus: pmc_abstracts_10k
Retriever: FAISS
Embedding model: /root/autodl-tmp/models/bge-small-en-v1.5
Reranker: keyword_reranker
Evidence top_k: 5
Candidate top_k: 10
Min rerank score: 0.0
```

Phase 3 has two completed full-scale RAG runs:

```text
Knowledge RAG:
  Experiment ID: exp03_rag_qwen_7b_4090d_pmc10k_full
  Prompt version: rag_v3
  Prompt contract: structured_rag_v3
  RAG mode: knowledge_acquisition
  Agent route: knowledge_agent -> retriever -> reranker -> evidence_filter -> clinical_reasoning_agent

Knowledge + Evidence RAG Verifier:
  Experiment ID: exp03_knowledge_evidence_rag_qwen_7b_4090d_pmc10k_full
  Prompt version: knowledge_evidence_rag_v1
  Prompt contract: structured_knowledge_then_evidence_verification_v1
  RAG mode: knowledge_then_claim_verification
  Agent route: knowledge_agent -> retriever -> reranker -> evidence_filter -> reasoning_agent -> claim_extractor -> evidence_agent -> retriever -> reranker -> evidence_filter -> verifier_agent
```

## Data Version

This experiment uses the project-wide VQA-RAD policy documented in
`Docs/vqa_rad_data_version_audit.md`.

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

## Main Metrics

Full-scale results on the 451-sample VQA-RAD test split:

```text
Knowledge RAG accuracy: 0.22838137472283815
Knowledge RAG mean token F1: 0.2962545638729584
Knowledge RAG mean BLEU-1: 0.2779809127631062
Knowledge RAG correct samples: 103 / 451
Knowledge RAG valid main result: true

Knowledge + Evidence RAG Verifier accuracy: 0.11529933481152993
Knowledge + Evidence RAG Verifier mean token F1: 0.1811753818464053
Knowledge + Evidence RAG Verifier mean BLEU-1: 0.16141674346441812
Knowledge + Evidence RAG Verifier correct samples: 52 / 451
Knowledge + Evidence RAG Verifier valid main result: true
```

The Knowledge RAG run is the stronger Phase 3 result by final-answer accuracy. The
Knowledge + Evidence verifier route retrieved more evidence and produced explicit
claim verification labels, but it reduced answer accuracy substantially.

Compared with the tracked Phase 1 Direct VLM baseline:

```text
Direct VLM accuracy: 0.49223946784922396
Direct VLM correct samples: 222 / 451
Knowledge RAG delta vs Direct VLM: -0.2638580931263858
Knowledge + Evidence RAG Verifier delta vs Direct VLM: -0.376940133037694
```

No full Phase 2 CoT metrics file was present under `Results/` at the time of this
summary, so the controlled CoT-without-RAG comparison should be added once the
Phase 2 full run is available.

## Evidence Metrics

```text
Knowledge RAG mean evidence quality score: 0.6071743192904656
Knowledge RAG mean question coverage: 0.6664704168514413
Knowledge RAG mean answer coverage: 0.334125
Knowledge RAG evidence empty rate: 0.0
Knowledge RAG retrieved evidence count: 5 for all 451 samples

Knowledge + Evidence RAG Verifier mean evidence quality score: 0.6832349490022173
Knowledge + Evidence RAG Verifier mean question coverage: 0.7237951441241686
Knowledge + Evidence RAG Verifier mean answer coverage: 0.495999995
Knowledge + Evidence RAG Verifier evidence empty rate: 0.0
Knowledge + Evidence RAG Verifier retrieved evidence count: 10 for all 451 samples
```

The verifier route improved the rule-based evidence-quality metrics, especially
answer coverage. This did not translate into better final answers, suggesting that
higher lexical evidence coverage alone is not sufficient for this architecture.

## Answer-Type Breakdown

```text
Knowledge RAG CLOSED questions: 91 / 272 correct = 0.33455882352941174
Knowledge RAG OPEN questions: 12 / 179 correct = 0.0670391061452514

Knowledge + Evidence RAG Verifier CLOSED questions: 41 / 272 correct = 0.15073529411764705
Knowledge + Evidence RAG Verifier OPEN questions: 11 / 179 correct = 0.061452513966480445
```

Both Phase 3 routes remain much stronger on closed questions than open questions.
Knowledge RAG improves over the verifier route mostly on closed questions, while
open-answer performance is low for both.

## Claim Verification Breakdown

```text
Knowledge RAG claim verification status:
NOT_APPLICABLE: 451

Knowledge + Evidence RAG Verifier claim verification status:
UNSUPPORTED: 285
SUPPORTED: 164
CONTRADICTED: 2
```

The verifier marked 63.19% of samples as `UNSUPPORTED`, 36.36% as `SUPPORTED`,
and 0.44% as `CONTRADICTED`. This conservative behavior likely suppressed many
answers, including answers that may have matched the VQA-RAD short ground truth.

## Latency

```text
Knowledge RAG average latency: 5717.17 ms
Knowledge RAG median latency: 5753.81 ms
Knowledge RAG P90 latency: 6566.21 ms
Knowledge RAG P95 latency: 6711.73 ms
Knowledge RAG max latency: 12967.65 ms

Knowledge + Evidence RAG Verifier average latency: 11395.60 ms
Knowledge + Evidence RAG Verifier median latency: 11636.56 ms
Knowledge + Evidence RAG Verifier P90 latency: 12849.89 ms
Knowledge + Evidence RAG Verifier P95 latency: 13168.81 ms
Knowledge + Evidence RAG Verifier max latency: 21695.92 ms
```

The slowest full-run sample was the same in both routes:

```text
sample_id: 10
image_id: synpic42202.jpg
question: Is there evidence of an aortic aneurysm?
Knowledge RAG latency_ms: 12967.649
Knowledge + Evidence RAG Verifier latency_ms: 21695.916
```

The verifier route roughly doubled median latency because it performs additional
claim extraction, evidence retrieval, reranking, filtering, and verification.

## Output Format Check

```text
Knowledge RAG empty outputs: 0
Knowledge RAG outputs longer than 10 words: 45

Knowledge + Evidence RAG Verifier empty outputs: 0
Knowledge + Evidence RAG Verifier outputs longer than 10 words: 60
```

Both runs produced non-empty predictions for all samples. However, both still have
some long final-answer strings, especially the verifier route. For VQA-RAD exact
or short-answer evaluation, later prompts should enforce shorter final answers
more aggressively.

## Smoke Runs

```text
Knowledge RAG smoke experiment: exp03_rag_qwen_7b_4090d_pmc10k_20
Samples: 20
Accuracy: 0.4
Mean token F1: 0.4
Mean BLEU-1: 0.4
Mean evidence quality score: 0.6616667
Correct samples: 8 / 20

Knowledge + Evidence RAG Verifier smoke experiment: exp03_knowledge_evidence_rag_qwen_7b_4090d_pmc10k_20
Samples: 20
Accuracy: 0.1
Mean token F1: 0.16642246642246644
Mean BLEU-1: 0.13874999999999998
Mean evidence quality score: 0.7216667
Correct samples: 2 / 20
```

The 20-sample smoke results predicted the direction of the full run: Knowledge RAG
was stronger on answer accuracy, while the verifier route scored higher on
evidence-quality metrics.

## Excluded / Non-Final Run

```text
Results/exp03_rag_qwen_7b_cuda_full/predictions.jsonl
Rows present: 45
Metrics file: missing
Accuracy computed from current prediction records: 0 / 45 = 0.0
Reason for exclusion: incomplete run without metrics.json and not comparable to the 451-sample full runs.
```

## Result Files

```text
Results/exp03_rag_qwen_7b_4090d_pmc10k_full/predictions.jsonl
Results/exp03_rag_qwen_7b_4090d_pmc10k_full/metrics.json
Results/exp03_knowledge_evidence_rag_qwen_7b_4090d_pmc10k_full/predictions.jsonl
Results/exp03_knowledge_evidence_rag_qwen_7b_4090d_pmc10k_full/metrics.json
Results/exp03_rag_qwen_7b_4090d_pmc10k_20/predictions.jsonl
Results/exp03_rag_qwen_7b_4090d_pmc10k_20/metrics.json
Results/exp03_knowledge_evidence_rag_qwen_7b_4090d_pmc10k_20/predictions.jsonl
Results/exp03_knowledge_evidence_rag_qwen_7b_4090d_pmc10k_20/metrics.json
```

`Results/` is ignored by Git, so this document is the tracked summary of the
Phase 3 experiments.

## Interpretation

Phase 3 successfully ran a full Medical RAG comparison on the same 451-sample
VQA-RAD test split. The main finding is negative for the current RAG design:
external retrieval did not improve over the Phase 1 Direct VLM baseline, and the
claim-verification route further reduced final-answer accuracy.

The strongest Phase 3 configuration is Knowledge RAG with FAISS retrieval over the
PMC 10k corpus. It achieved 103 / 451 correct answers, but this is still well
below the Direct VLM baseline of 222 / 451. The likely failure mode is that
retrieved abstracts often provide generic medical context rather than image- and
question-specific evidence. The model then produces cautious or verbose answers
that do not match VQA-RAD's short ground-truth labels.

The Knowledge + Evidence verifier route is useful diagnostically because it records
claim-level verification status, knowledge evidence, verification evidence,
`generated_claims`, `verified_claim`, and `critic_decision`. As a scoring method,
however, it is currently too conservative: most claims are marked unsupported, and
accuracy falls to 52 / 451.

Next Phase 3 work should focus on retrieval quality and answer normalization before
building more complex multi-agent routing. Recommended next steps are:

1. Add image-finding-aware retrieval queries instead of question-only retrieval.
2. Enforce short final answers for closed VQA-RAD questions.
3. Separate evidence-quality metrics from final-answer scoring in the main table.
4. Compare against a completed full Phase 2 CoT run using the same 451-sample split.
5. Inspect false negatives where verifier output is semantically correct but too cautious for exact VQA-RAD labels.

# Phase 1 Direct VLM Baseline Results

## Experiment

```text
Experiment ID: exp01_direct_vlm_qwen_7b_cuda_full
Method: Direct VLM inference
Backbone: Qwen/Qwen2.5-VL-7B-Instruct
Backend: qwen2_5_vl
Training epochs: 0
Prompt version: direct_v1
Dataset: VQA-RAD OSF public release
Split: test
Scale: full
Number of test samples: 451
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

```text
Accuracy: 0.49223946784922396
Mean token F1: 0.5541281999596854
Mean BLEU-1: 0.5397320417399463
Correct samples: 222 / 451
Valid main result: true
```

## Answer-Type Breakdown

```text
CLOSED questions: 192 / 272 correct = 0.7058823529411765
OPEN questions: 30 / 179 correct = 0.16759776536312848
```

The direct baseline is much stronger on closed yes/no-style questions than on
open descriptive questions. This is an important motivation for Phase 2
structured prompting and Chain-of-Thought experiments.

## Latency

```text
Average latency: 4357.90 ms
Median latency: 2568.68 ms
P90 latency: 7441.99 ms
P95 latency: 9901.45 ms
Max latency: 211786.60 ms
```

The slowest sample was:

```text
sample_id: 10
image_id: synpic42202.jpg
question: Is there evidence of an aortic aneurysm?
latency_ms: 211786.60
```

Latency has outliers, so future reporting should include median and percentile
latency, not only mean latency.

## Output Format Check

```text
Empty outputs: 0
Outputs longer than 10 words: 12
```

The prompt usually produced short answers, but some open questions still caused
descriptive sentence outputs. This should be handled in later prompt/schema
experiments rather than changing the Phase 1 direct baseline after the fact.

## Result Files

```text
Results/exp01_direct_vlm_qwen_7b_cuda_full/predictions.jsonl
Results/exp01_direct_vlm_qwen_7b_cuda_full/metrics.json
```

`Results/` is ignored by Git, so this document is the tracked summary of the
Phase 1 full baseline.

## Interpretation

Direct VLM inference establishes the baseline performance for the controlled
experiment chain. The main weakness is open-answer medical description, where
the model only achieved 30 correct answers out of 179 open questions. This
supports the next research step: test whether structured reasoning prompts and
Chain-of-Thought can improve open-ended reasoning without changing the frozen
backbone.

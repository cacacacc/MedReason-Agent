# Experiment Protocol

All main experiments should follow this protocol unless the experiment explicitly studies one variable.

## Fixed Variables

- Backbone: one main VLM for the controlled chain.
- Dataset split: fixed per dataset.
- Prompt version: versioned and logged.
- Image preprocessing: fixed and logged.
- Generation settings: deterministic where possible.
- Evaluation scripts: same script for comparable experiments.

## Default Model Policy

Main controlled experiments:

- Qwen2.5-VL-7B-Instruct

Development fallback:

- Qwen2.5-VL-3B-Instruct

External baseline only:

- LLaVA-Med

Do not mix different backbones inside the main controlled comparison chain.

## Default Generation Settings

```yaml
temperature: 0.0
top_p: 1.0
max_new_tokens: 1024
seeds:
  - 42
  - 2024
  - 3407
```

If a provider does not support deterministic seeds, record that limitation in the run metadata.

## Dataset Scale Policy

Every new pipeline must pass three scales before full evaluation:

```text
Smoke Test: 10-20 samples
Dev Experiment: 100-200 samples
Intermediate Experiment: 500-1000 samples
Full Experiment: stable pipeline only
```

## Main Experiment Chain

```text
Experiment 0: Random / Majority Baseline
Experiment 1: Direct VLM
Experiment 2: Structured Direct Prompt
Experiment 3: Chain-of-Thought
Experiment 4: Self-Consistency on small subset
Experiment 5: RAG-Augmented Reasoning
Experiment 6: Fixed Multi-Agent
Experiment 7: Supervisor Multi-Agent
Experiment 8: Supervisor + Verification
Experiment 9: Adaptive Reasoning
Experiment 10: Optional LoRA
```

## Required Result Fields

Every prediction record should include:

```json
{
  "experiment_id": "",
  "model": "",
  "prompt_version": "",
  "dataset": "",
  "split": "",
  "sample_id": "",
  "image_id": "",
  "question": "",
  "prediction": "",
  "ground_truth": "",
  "reasoning_output": "",
  "retrieved_evidence": [],
  "tool_calls": [],
  "agent_route": [],
  "critic_decision": "",
  "confidence": null,
  "latency_ms": null,
  "input_tokens": null,
  "output_tokens": null,
  "correct": null,
  "error_type": ""
}
```

## Reporting

Report:

- Mean and standard deviation when multiple runs are available.
- Accuracy and task-specific answer metrics.
- Reliability metrics.
- Efficiency metrics.
- Error taxonomy breakdown.

Do not rely on terminal output as the final experiment record. Save structured JSON / CSV files.

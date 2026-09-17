# Vision Consistency Agent

## Motivation

The current multi-agent pipeline can suffer from early visual noise: if the
Vision Agent produces an unstable or over-specific observation, the Reasoning
Agent may treat that observation as fact and propagate the error.

Inspired by MedMMV's idea of stabilizing early evidence interpretation with
diversified short rollouts, this project adds a lightweight VQA-specific
variant:

```text
Complex sample
  -> Vision observation pass 1
  -> Vision observation pass 2
  -> Vision observation pass 3
  -> Vision consensus extraction
  -> Reasoning Agent
```

This is not a full MedMMV reimplementation. It does not build a complete
multimodal evidence graph or uncertainty scorer. It only stabilizes the visual
facts before they enter the Reasoning Agent.

## When It Runs

The feature is controlled by:

```yaml
method:
  vision_consistency_enabled: true
  vision_observation_count: 3
```

It only triggers for complex visual samples, including:

- open-answer questions;
- ABN / OTHER / SIZE / ATTRIB / POS question types;
- questions containing markers such as abnormality, lesion, mass, opacity,
  consolidation, aneurysm, fracture, edema, infarct, side comparison, or size
  comparison.

Simple closed visual questions still use the original single-pass Vision Agent.

## Output Fields

Prediction records now include:

```text
vision_observations
vision_consensus
agent_outputs.vision_observations
agent_outputs.vision_consensus
```

`agent_outputs.vision` is the text passed to downstream agents. When consistency
is enabled for a complex sample, it contains the consensus visual facts rather
than a single raw observation.

Tool calls record:

```text
vision_observation
vision_consensus
```

## Controlled Ablation

Baseline:

```bash
python scripts/run_multi_agent.py \
  --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_100.yaml
```

Vision consistency:

```bash
python scripts/run_multi_agent.py \
  --config configs/experiments/exp04_supervisor_multi_agent_vision_consistency_qwen_7b_4090d_pmc10k_100.yaml
```

Compare:

- accuracy;
- closed vs open accuracy;
- hallucination rate;
- evidence quality;
- mean latency;
- input/output tokens;
- error attribution;
- whether incorrect complex samples improve without hurting simple samples.

## Expected Trade-off

The likely benefit is reduced early visual noise on complex samples.

The cost is additional VLM calls. A 3-pass complex sample uses:

```text
3 observation calls + 1 consensus call
```

Therefore this should be evaluated as an accuracy/reliability vs latency/cost
trade-off, not as a free improvement.

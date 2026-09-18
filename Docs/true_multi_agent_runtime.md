# Contract-Based Multi-Agent Runtime

## Motivation

The earlier `supervisor_multi_agent` pipeline used role-specific prompts and a
shared state, but most specialist roles were still executed as inline
`backend.generate()` calls inside one controller. This was useful for controlled
experiments, but it left an interview risk: the implementation could be
described as a role-prompted multi-stage pipeline rather than a true multi-agent
runtime.

The new `agent_runtime_multi_agent` mode makes the agent boundary explicit while
keeping the original model, prompts, retrieval pipeline, metrics, and output
schema comparable.

## Architecture

The runtime is implemented in:

- `src/medreason_agent/agents/true_multi_agent.py`
- `src/medreason_agent/agents/multi_agent.py`

Each role is now represented by a separate agent object:

- `SupervisorAgent`: selects the specialist tools.
- `VisionAgent`: writes visual observations and visual claim statuses.
- `RetrievalAgent`: has exclusive access to the retrieval pipeline.
- `ReasoningAgent`: converts observations and evidence into hypotheses.
- `VerifierAgent`: verifies claims against a fresh retrieval pass.
- `AnswerAgent`: produces the final benchmark answer from committed state.

Each agent returns an `AgentStepResult` with a structured contract:

- `agent_name`
- `raw_output`
- `parsed_output`
- `claim_statuses`
- `tool_calls`
- `retrieved_evidence`
- token/confidence metadata

The `AgentRuntimeMultiAgent` executor commits each result to
`SharedAgentState`. Agents communicate through committed state rather than by
directly passing arbitrary raw prompt text.

## What Changed

Before:

```text
SupervisorMultiAgent.run()
  -> build prompt
  -> backend.generate()
  -> mutate local variables
  -> write shared state
```

After:

```text
AgentRuntimeMultiAgent.run()
  -> SupervisorAgent.run(context)
  -> SupervisorPolicy.build_plan(...)
  -> SpecialistAgent.run(context)
  -> runtime commits AgentStepResult to SharedAgentState
  -> AnswerAgent.run(context)
```

This is still a single-backbone experiment: all VLM-backed agents can share the
same frozen Qwen2.5-VL model for controlled comparison. The multi-agent claim is
now based on runtime structure, agent contracts, state commits, and tool
boundaries, not only on different prompt names.

## Controlled Experiment

Run the old supervisor workflow:

```bash
python scripts/run_multi_agent.py \
  --config configs/experiments/exp04_supervisor_multi_agent_qwen_7b_4090d_pmc10k_100.yaml
```

Run the contract-based runtime:

```bash
python scripts/run_multi_agent.py \
  --config configs/experiments/exp04_agent_runtime_multi_agent_qwen_7b_4090d_pmc10k_100.yaml
```

For this ablation, keep these variables fixed:

- dataset split and `max_samples`
- Qwen2.5-VL model
- prompt family
- FAISS/BGE retriever
- reranker
- `top_k` and `candidate_top_k`

Only `agent_mode` changes:

- baseline: `supervisor_multi_agent`
- improved runtime: `agent_runtime_multi_agent`

## Interview-Safe Description

An accurate description is:

> I first implemented a supervisor-guided role-prompt workflow. I then refactored
> it into a contract-based multi-agent runtime where each role is a separate
> agent object with a structured input/output contract, controlled tool access,
> shared-state commits, and auditable tool-call traces. The same frozen VLM is
> shared across agents to keep the ablation controlled.

Avoid claiming that the system trains multiple independent models. The
contribution is the multi-agent runtime structure and controlled evaluation, not
multi-model training.

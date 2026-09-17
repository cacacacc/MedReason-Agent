# Research Questions

This document defines what MedReason-Agent is trying to answer. It is written as
a research map rather than a feature list: every module, experiment, and resume
claim should connect back to one of these questions.

## Project Purpose

MedReason-Agent studies whether a multimodal medical VQA system can become more
reliable by changing its reasoning workflow, retrieval support, routing policy,
and lightweight adaptation strategy while keeping the backbone model controlled.

The project is not a clinical diagnosis product. It is a benchmark research
project on VQA-RAD using Qwen2.5-VL as the main vision-language backbone.

The central problem is:

```text
Given a medical image and a natural-language question, how should a frozen or
lightly adapted VLM decide whether to answer directly, reason step by step,
retrieve external medical evidence, invoke verifier agents, or route through a
more complex multi-agent workflow?
```

The project's most important finding so far is not "more agents always help."
The evidence points in the opposite direction:

```text
Reliability improves when the system avoids unnecessary reasoning depth and
selectively uses structured reasoning only where it helps.
```

## Main Research Question

Can adaptive reasoning control improve multimodal medical VQA accuracy and
efficiency compared with direct VLM inference, Chain-of-Thought prompting,
RAG-augmented reasoning, and fixed multi-agent pipelines?

The project measures "reliability" with:

- answer accuracy / exact match;
- token F1 and BLEU-1;
- evidence quality and evidence coverage;
- hallucination and unsupported-claim indicators;
- latency, token usage, and agent call count;
- route selection quality and oracle-routing gap.

## RQ1: Does Explicit Reasoning Improve Medical VQA?

Question:

Does Chain-of-Thought or structured reasoning improve VQA-RAD performance when
the backbone remains Qwen2.5-VL-7B?

Experiment:

- Direct VLM baseline.
- Vanilla CoT.
- Structured CoT.
- Same dataset split, model, and evaluation pipeline.

Current answer:

No, not as a universal strategy. Full-test Phase 2 results show both CoT variants
underperformed direct answering:

```text
Direct Answer: 49.89%
Vanilla CoT: 43.68%
Structured CoT: 43.90%
```

Interpretation:

CoT can help some question types, but it also introduces reasoning drift. The
project should not claim that CoT alone improves medical VQA. The stronger claim
is that structured reasoning must be selectively routed.

## RQ2: Does External Medical Retrieval Improve Answers?

Question:

Does adding PubMed/PMC evidence through RAG improve answer accuracy or evidence
grounding?

Experiment:

- Knowledge RAG: retrieve evidence before reasoning.
- Knowledge + Evidence RAG: retrieve knowledge, generate claims, retrieve
  verification evidence, and verify claims.
- Current main vector setup: BGE-small embeddings + FAISS over PMC/PubMed 10k
  abstracts.

Current answer:

RAG improved traceability and evidence records, but it did not improve final
answer accuracy over direct VLM in the completed Phase 3 runs.

```text
Direct VLM: 222 / 451 = 49.22%
Knowledge RAG: 103 / 451 = 22.84%
Knowledge + Evidence RAG: 52 / 451 = 11.53%
```

Interpretation:

The current retriever often supplies generic medical context rather than
image-specific evidence. More evidence and higher lexical evidence coverage do
not automatically produce better short VQA answers.

## RQ3: Do Fixed Multi-Agent Pipelines Improve Reliability?

Question:

Does splitting the task into specialist agents, such as vision, reasoning,
critic/verifier, retrieval, and answer agents, improve performance?

Experiment:

- Fixed Multi-Agent.
- Supervisor Multi-Agent with retrieval and verifier.
- Shared state and state compression are recorded for auditability.

Current answer:

No, not when applied broadly. Completed Phase 4 full-test results showed that
fixed and supervisor multi-agent pipelines underperformed direct VLM.

```text
Direct VLM: about 49-50%
Fixed Multi-Agent: about 39-40%
Supervisor Multi-Agent + PMC 10k: about 31%
```

Interpretation:

Multi-agent structure is useful for audit fields and error analysis, but it can
increase answer drift and cost. The project therefore moved from "always use
more agents" to "route only when the sample needs it."

## RQ4: Does Verification or Self-Reflection Correct Errors?

Question:

Can self-reflection or a separate verifier correct wrong answers without causing
too many regressions?

Experiment:

- Supervisor No Critic.
- Self-Reflection.
- Separate Verifier.
- Candidate-to-final answer transition metrics.

Current answer:

Not yet. Phase 5 showed that verification modules can record process-level
signals, but they did not improve final answer accuracy in the current prompt
and routing setup.

```text
Supervisor No Critic: 21.73%
Self-Reflection: 17.29%
Separate Verifier: 20.18%
```

Interpretation:

Verification currently works better as an audit mechanism than as a reliable
answer-correction mechanism. It can identify unsupported claims, but converting
that signal into better short answers remains unresolved.

## RQ5: Is Adaptive Reasoning Better Than Always-Complex Reasoning?

Question:

Can the system improve accuracy and efficiency by routing simple questions to
direct answering and only sending selected cases to structured reasoning or
multi-agent workflows?

Experiment:

- Rule-based adaptive routing v1-v6.
- Direct, Structured CoT, and Full Multi-Agent candidate routes.
- Oracle-routing analysis to estimate the best route distribution.

Current answer:

Yes. This is the strongest completed research direction in the project. Phase 6
showed that reducing over-reasoning improves both accuracy and efficiency.

Key full-test result:

```text
Phase 6 v6 Adaptive Routing: 57.43%
Mean agent calls: 1.00
Full Multi-Agent route: disabled in v6
```

Interpretation:

The best current strategy is Direct/Structured-CoT dominant routing, not a
heavy multi-agent route. The supervisor's value is choosing reasoning depth,
especially deciding when not to use complex agents.

## RQ6: Does Lightweight Domain Adaptation Improve the Backbone?

Question:

Can LoRA fine-tuning improve the base VQA ability enough to raise the ceiling of
the adaptive-routing system?

Experiment:

- Qwen2.5-VL LoRA SFT on VQA-RAD train/validation.
- Short-answer training objective.
- Evaluation through Direct VLM and Phase 6 routing configs.

Current answer:

LoRA improved direct VQA performance over the frozen baseline in the tracked
Phase 8 results.

```text
Frozen Direct: about 49-50%
Frozen Phase 6 v6: 57.43%
LoRA Direct: 59.87%
LoRA Phase 6 v7: 58.76%
```

Interpretation:

LoRA can improve the base model's short-answer ability. However, routing rules
must be recalibrated after fine-tuning, because a stronger direct model changes
when structured reasoning is useful.

## RQ7: Can LangGraph and Qdrant Improve the System?

Question:

Do explicit graph orchestration and vector-database retrieval improve the
system compared with the existing Python orchestration and FAISS retrieval?

Current implementation status:

- LangGraph orchestration has been added as an optional RAG orchestrator.
- Qdrant retrieval has been added as an optional retriever.
- Controlled ablation configs now exist for:
  - Python + FAISS baseline;
  - LangGraph + FAISS;
  - Python + Qdrant;
  - LangGraph + Qdrant.

Current answer:

Not answered yet. This is now an implemented future experiment, not a completed
result.

Expected interpretation:

- LangGraph should mainly improve explicit state transitions, auditability, and
  future branching/retry logic. It should not be expected to increase accuracy
  by itself if prompts, model calls, and retrieved evidence are unchanged.
- Qdrant should mainly test vector-store infrastructure and extensibility. On a
  10k corpus with the same BGE embeddings, it may not outperform FAISS on
  accuracy, but it is useful for scalable retrieval experiments.

See `Docs/langgraph_qdrant_ablation.md`.

## Overall Answer So Far

The project has answered a negative result and a positive result.

Negative result:

```text
Naive CoT, broad RAG, fixed multi-agent pipelines, and unconditional verifier
loops do not reliably improve VQA-RAD final-answer accuracy.
```

Positive result:

```text
Adaptive reasoning depth is useful. The strongest frozen-backbone result comes
from mostly direct answering plus selective structured CoT, not from always
using complex multi-agent reasoning.
```

Research contribution:

```text
The project reframes medical VQA reliability as a routing and reasoning-control
problem: the key question is not "how many agents can we add?", but "which
samples actually need extra reasoning, retrieval, or verification?"
```

## Interview-Safe Summary

If asked "what is your project about?", answer:

```text
My project studies reliable multimodal medical VQA with Qwen2.5-VL on VQA-RAD.
I built a controlled experiment pipeline comparing direct answering, CoT, RAG,
multi-agent reasoning, verifier loops, adaptive routing, and LoRA adaptation.
The main finding is that adding more reasoning modules does not automatically
help. Broad CoT, RAG, and multi-agent routes often reduce accuracy. The best
frozen-backbone result comes from adaptive routing: use direct answering for
simple cases and structured CoT only for selected questions. This improved
accuracy while keeping the average agent-call cost low.
```

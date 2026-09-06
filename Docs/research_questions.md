# Research Questions

This document turns the project idea into experiment-testable research questions.

## Main Question

Can a supervisor-guided multi-agent reasoning framework improve multimodal medical VQA reliability compared with direct VLM inference?

## RQ1: Explicit Reasoning

Question:

Can explicit reasoning improve medical multimodal question answering compared with direct VLM inference?

Operational test:

- Compare Direct VLM vs Chain-of-Thought vs structured reasoning prompt.
- Use the same backbone, dataset split, preprocessing, and decoding settings.

Primary metrics:

- Accuracy / exact match
- Open-ended answer match after normalization

Secondary metrics:

- Reasoning consistency
- Output token cost
- Latency

## RQ2: Multi-Agent Collaboration

Question:

Can multi-agent collaboration improve multimodal medical reasoning compared with a single VLM or simple CoT?

Operational test:

- Compare Direct VLM, CoT, and Fixed Multi-Agent.

Key risk:

- More agents can introduce more errors, longer context, and inconsistent intermediate claims.

## RQ3: Supervisor Routing

Question:

Can a Supervisor Agent dynamically route reasoning tasks more effectively than a fixed multi-agent pipeline?

Operational test:

- Compare Fixed Multi-Agent vs Supervisor Multi-Agent.
- Keep available agents and prompts controlled as much as possible.

Metrics:

- Accuracy
- Average agent calls
- Average steps
- Latency
- Token usage

## RQ4: Verification

Question:

Can a Verification / Critic Agent reduce hallucination and correct faulty reasoning?

Operational test:

- Compare Supervisor without Critic vs Supervisor with Critic.
- Track both correction and regression.

Metrics:

- Error correction rate
- Correct-to-wrong revision rate
- Unsupported claim rate
- Hallucination rate

## RQ5: Adaptive Reasoning

Question:

Can adaptive reasoning improve the accuracy-efficiency trade-off by avoiding expensive multi-agent reasoning on simple questions?

Operational test:

- Compare Always Full Reasoning vs Adaptive Routing.

Metrics:

- Accuracy
- Token usage
- Latency
- Cost per task

## RQ6: Calibration

Question:

Are model confidence estimates calibrated, and can verification improve calibration?

Operational test:

- Collect confidence scores from each method.
- Compare confidence against correctness.

Metrics:

- Expected Calibration Error
- Brier Score
- Reliability diagrams

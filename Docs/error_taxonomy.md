# Error Taxonomy

This taxonomy is the first version. It will be updated after real error analysis.

## Error Types

`VISUAL_PERCEPTION_ERROR`

The model misreads the image or misses a visible finding.

`KNOWLEDGE_ERROR`

The model lacks or misuses medical knowledge.

`RETRIEVAL_ERROR`

The retrieval system returns irrelevant, misleading, or insufficient evidence.

`REASONING_ERROR`

The model has the needed information but reaches the wrong answer.

`LOGICAL_LEAP`

The conclusion goes beyond what the image, question, or evidence supports.

`HALLUCINATION`

The answer invents findings, patient details, or evidence not present in the input.

`SUPERVISOR_ROUTING_ERROR`

The supervisor chooses the wrong path, such as skipping retrieval for a knowledge-heavy question.

`TOOL_SELECTION_ERROR`

The system selects the wrong tool or uses the right tool at the wrong time.

`VERIFICATION_ERROR`

The verifier fails to catch an error or incorrectly rejects a valid answer.

`CORRECT_TO_WRONG_REVISION`

The initial answer is correct, but the critic or revision step changes it to an incorrect answer.

`FINAL_ANSWER_FORMAT_ERROR`

The final answer is semantically present but violates the required format.

`UNKNOWN`

The error cannot be confidently assigned.

## Error Analysis Rule

After each complete experiment, sample incorrect predictions and classify errors. Do not change the system based only on intuition.

Example report:

```text
100 incorrect samples:
35% visual perception
30% reasoning
15% retrieval
10% supervisor routing
10% other
```

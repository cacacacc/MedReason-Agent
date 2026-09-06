# MedReason-Agent

> A Supervisor-Guided Multi-Agent Framework for Reliable Multimodal Medical Reasoning

MedReason-Agent is a research-oriented AI project about multimodal medical visual question answering. The goal is not to build a clinical diagnosis product. The goal is to test whether explicit reasoning, retrieval, multi-agent collaboration, supervisor routing, and verification can improve the reliability of a frozen vision-language model on medical VQA benchmarks.

## Research Focus

The main research question is:

Can a supervisor-guided multi-agent reasoning framework improve accuracy, evidence grounding, calibration, and efficiency compared with direct VLM inference?

The project will compare a controlled sequence of systems:

1. Random / majority baseline
2. Direct VLM
3. Structured direct prompt
4. Chain-of-thought prompting
5. RAG-augmented reasoning
6. Fixed multi-agent pipeline
7. Supervisor multi-agent routing
8. Supervisor with verification / reflection
9. Adaptive reasoning
10. Optional LoRA fine-tuning

## Core Principle

Main experiments use a frozen backbone. Improvements should come from reasoning architecture and evaluation design, not from changing model size between experiments.

Default backbone:

- Main: Qwen2.5-VL-7B-Instruct
- Lightweight development fallback: Qwen2.5-VL-3B-Instruct

## Current Phase

Phase 0: Research Foundation + Repository Setup

Current work:

- Define research questions.
- Define controlled experiment protocol.
- Define first error taxonomy.
- Build learning materials for the project fundamentals.

No core agent code is implemented yet.

## Initial Structure

```text
MedReason-Agent/
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── exp00_majority.yaml
│       └── exp01_direct_vlm.yaml
├── Data/
│   ├── Raw/
│   └── Processed/
├── Docs/
│   ├── research_questions.md
│   ├── experiment_protocol.md
│   └── error_taxonomy.md
├── Experiments/
├── Results/
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   ├── agent.txt
│   ├── rag.txt
│   └── vlm.txt
├── scripts/
│   └── check_environment.py
├── src/
│   └── medreason_agent/
├── assets/
│   └── lesson.css
├── lessons/
│   └── 0001-medical-vqa-and-baseline.html
├── MISSION.md
├── NOTES.md
├── RESOURCES.md
└── README.md
```

## Medical Safety Boundary

All outputs in this repository should be described as model predictions for benchmark research. They are not medical advice, clinical diagnosis, or a substitute for professional medical judgment.

## Environment Setup

Recommended local setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements/dev.txt
```

Optional dependency groups:

```powershell
pip install -r requirements/agent.txt
pip install -r requirements/rag.txt
pip install -r requirements/vlm.txt
```

Check the environment:

```powershell
python scripts/check_environment.py
pytest -q
```

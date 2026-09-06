# MedReason-Agent Resources

## Knowledge

- [VQA-RAD: A dataset of clinically generated visual questions and answers about radiology images](https://pmc.ncbi.nlm.nih.gov/articles/PMC6244189/)
  Foundational dataset paper for the main Medical VQA benchmark. Use for: dataset motivation, question types, clinical validation, and evaluation design.

- [Medical visual question answering: A survey](https://pubmed.ncbi.nlm.nih.gov/37673579/)
  Survey of Medical VQA datasets, methods, and challenges. Use for: Phase 0 literature grounding and related work structure.

- [Qwen2.5-VL Technical Report](https://huggingface.co/papers/2502.13923)
  Technical report for the default VLM backbone family. Use for: backbone selection, model capability description, and limitations.

- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://mlanthology.org/neurips/2023/li2023neurips-llavamed/)
  Biomedical VLM reference model. Use for: medical multimodal model background and possible external baseline discussion.

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://huggingface.co/papers/2201.11903)
  Foundational CoT prompting paper. Use for: reasoning baseline motivation and limitations.

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/)
  Agent reasoning-and-action framework. Use for: agent loop, tool use, and retrieval action design.

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://doi.org/10.48550/arXiv.2303.11366)
  Reflection-based agent improvement without weight updates. Use for: verification, critique, and self-correction design.

- [PMC-VQA: Development of a large-scale medical visual question-answering dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC11663219/)
  Large-scale Medical VQA dataset paper. Use for: external validation planning after the VQA-RAD pipeline is stable.

## Wisdom (Communities)

- [Papers with Code: VQA-RAD](https://paperswithcode.com/dataset/vqa-rad)
  Use for: checking public baselines and dataset references.

- [Hugging Face Papers](https://huggingface.co/papers)
  Use for: tracking model papers, implementations, and community discussions.

## Gaps

- Need to verify specific 2024-2026 papers for medical multi-agent reasoning, supervisor routing, multimodal RAG, and calibration before writing Related Work.
- Need to choose the exact VQA-RAD source package and split policy before Phase 1 implementation.

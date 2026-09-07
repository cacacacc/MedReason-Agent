"""构建 Phase 3 RAG 使用的本地医学知识库。

第一版支持两种来源：

1. `--seed-medical-vqa`：生成一个很小的教学/冒烟测试知识库，只用于验证 pipeline。
2. `--input-jsonl`：读取真实医学摘要 JSONL，切分成 chunks 后写出。

真实主实验应该使用 PubMed/PubMed Central 摘要生成的 JSONL，而不是 seed 知识库。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medreason_agent.paths import resolve_project_path
from medreason_agent.retrieval.chunking import KnowledgeDocument, chunk_documents
from medreason_agent.retrieval.keyword import write_chunks


SEED_DOCUMENTS = [
    KnowledgeDocument(
        doc_id="seed_chest_opacity",
        title="Chest radiograph opacity and consolidation",
        text=(
            "Airspace consolidation on chest radiographs can appear as increased pulmonary "
            "opacity. It may obscure vessels and can be associated with infection, edema, "
            "atelectasis, or other causes. Image findings must be interpreted with clinical context."
        ),
        source="seed_medical_vqa",
    ),
    KnowledgeDocument(
        doc_id="seed_aortic_aneurysm",
        title="Aortic aneurysm imaging findings",
        text=(
            "An aortic aneurysm is abnormal dilation of the aorta. On radiology images, clues "
            "may include widening of the aortic contour, mural calcification, or enlarged aortic "
            "diameter depending on modality and projection."
        ),
        source="seed_medical_vqa",
    ),
    KnowledgeDocument(
        doc_id="seed_small_bowel_obstruction",
        title="Small bowel obstruction imaging findings",
        text=(
            "Small bowel obstruction can show dilated small bowel loops, air-fluid levels, and "
            "a transition point. Contrast studies may help identify bowel loops and obstruction."
        ),
        source="seed_medical_vqa",
    ),
    KnowledgeDocument(
        doc_id="seed_brain_vascular_pathology",
        title="Brain infarction and vascular pathology",
        text=(
            "Brain infarction may appear as regions of abnormal attenuation or signal depending "
            "on imaging modality. Vascular pathology cannot be diagnosed from text alone and must "
            "be grounded in visible image findings."
        ),
        source="seed_medical_vqa",
    ),
]


def load_documents_from_jsonl(path: Path) -> list[KnowledgeDocument]:
    """从真实医学摘要 JSONL 读取文档。

    每行可以包含 `doc_id`/`pmid`、`title`、`abstract`/`text`、`source` 字段。
    这样脚本能适配不同 PubMed 导出格式。
    """
    documents: list[KnowledgeDocument] = []
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            doc_id = str(record.get("doc_id") or record.get("pmid") or f"doc_{index:06d}")
            title = str(record.get("title") or "")
            text = str(record.get("abstract") or record.get("text") or "")
            source = str(record.get("source") or path.name)
            if text.strip():
                documents.append(
                    KnowledgeDocument(
                        doc_id=doc_id,
                        title=title,
                        text=text,
                        source=source,
                    )
                )
    return documents


def parse_args() -> argparse.Namespace:
    """解析知识库构建参数。"""
    parser = argparse.ArgumentParser(description="构建 Phase 3 RAG 医学知识库。")
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--seed-medical-vqa", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/Processed/medical_kb/seed_chunks.jsonl"),
    )
    parser.add_argument("--chunk-size", type=int, default=120)
    parser.add_argument("--overlap", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed_medical_vqa:
        documents = SEED_DOCUMENTS
    elif args.input_jsonl is not None:
        documents = load_documents_from_jsonl(resolve_project_path(args.input_jsonl))
    else:
        raise ValueError("Use --seed-medical-vqa or provide --input-jsonl.")

    chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    output_path = resolve_project_path(args.output)
    write_chunks(output_path, chunks)

    print(
        json.dumps(
            {
                "num_documents": len(documents),
                "num_chunks": len(chunks),
                "output": str(output_path.relative_to(resolve_project_path("."))),
                "is_seed_kb": bool(args.seed_medical_vqa),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

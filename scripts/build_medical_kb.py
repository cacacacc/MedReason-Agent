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
            "atelectasis, or other causes. Image findings must be interpreted with "
            "clinical context."
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


def load_documents_from_jsonl(
    path: Path,
    max_documents: int | None = None,
) -> list[KnowledgeDocument]:
    """从真实医学摘要 JSONL 读取文档。

    每行可以包含 `doc_id`/`pmid`、`title`、`abstract`/`text`、`source` 字段。
    这样脚本能适配不同 PubMed / PubMed Central 导出格式。

    `max_documents` 用于控制第一版知识库规模，例如只取 10,000 篇有效摘要。
    这里按“有效摘要”计数：没有正文的记录不会占用 quota。
    """
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive when provided.")

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
                if max_documents is not None and len(documents) >= max_documents:
                    break
    return documents


def parse_args() -> argparse.Namespace:
    """解析知识库构建参数。"""
    parser = argparse.ArgumentParser(description="构建 Phase 3 RAG 医学知识库。")
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--seed-medical-vqa", action="store_true")
    parser.add_argument("--corpus-name", default="seed_medical_vqa")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument(
        "--chunk-unit",
        choices=["word", "tokenizer"],
        default="word",
        help="word 使用空格词切分；tokenizer 使用 Hugging Face tokenizer 切分。",
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="chunk-unit=tokenizer 时使用的 tokenizer 名称或本地路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/Processed/medical_kb/seed_chunks.jsonl"),
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=50)
    return parser.parse_args()


def load_tokenizer(name_or_path: str):
    """加载 Hugging Face tokenizer。

    放在函数内部懒加载，避免只跑 seed smoke 或单元测试时强制依赖 transformers。
    AutoDL 上如果已经上传 Qwen，本参数可以直接传本地模型目录。
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Tokenizer chunking requires transformers. Install with: "
            "pip install -r requirements/vlm.txt"
        ) from exc

    return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)


def main() -> int:
    args = parse_args()
    if args.seed_medical_vqa:
        documents = SEED_DOCUMENTS
    elif args.input_jsonl is not None:
        documents = load_documents_from_jsonl(
            resolve_project_path(args.input_jsonl),
            max_documents=args.max_documents,
        )
    else:
        raise ValueError("Use --seed-medical-vqa or provide --input-jsonl.")

    tokenizer = (
        load_tokenizer(args.tokenizer_name_or_path)
        if args.chunk_unit == "tokenizer"
        else None
    )
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        tokenizer=tokenizer,
    )
    output_path = resolve_project_path(args.output)
    write_chunks(output_path, chunks)

    print(
        json.dumps(
            {
                "num_documents": len(documents),
                "num_chunks": len(chunks),
                "output": str(output_path.relative_to(resolve_project_path("."))),
                "corpus_name": args.corpus_name,
                "max_documents": args.max_documents,
                "chunk_unit": args.chunk_unit,
                "tokenizer_name_or_path": (
                    args.tokenizer_name_or_path if args.chunk_unit == "tokenizer" else None
                ),
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "is_seed_kb": bool(args.seed_medical_vqa),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json

from scripts.build_medical_kb import load_documents_from_jsonl


def test_load_documents_from_jsonl_respects_max_documents(tmp_path) -> None:
    path = tmp_path / "pubmed_sample.jsonl"
    records = [
        {"pmid": "1", "title": "No abstract", "abstract": ""},
        {"pmid": "2", "title": "Chest opacity", "abstract": "Opacity can be seen."},
        {"pmid": "3", "title": "Aortic aneurysm", "abstract": "Aneurysm can dilate."},
        {"pmid": "4", "title": "Brain infarct", "abstract": "Infarct changes vary."},
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    documents = load_documents_from_jsonl(path, max_documents=2)

    assert [document.doc_id for document in documents] == ["2", "3"]
    assert len(documents) == 2


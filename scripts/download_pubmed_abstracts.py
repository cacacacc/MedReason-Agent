"""下载 PubMed 医学摘要并保存为 Phase 3 RAG 原始 JSONL。

输出文件默认是：

Data/Raw/pmc/pmc_abstracts.jsonl

注意：这里下载的是 PubMed abstracts，不是 PMC full text。Phase 3 第一版 RAG corpus
只需要 10,000 篇医学影像相关摘要，摘要库比全文库更轻，更适合先做 controlled
experiment。
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from tqdm import tqdm

from medreason_agent.paths import resolve_project_path

NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

DEFAULT_QUERY = (
    "("
    "radiology[Title/Abstract] OR imaging[Title/Abstract] OR "
    "x-ray[Title/Abstract] OR radiograph[Title/Abstract] OR "
    "CT[Title/Abstract] OR MRI[Title/Abstract] OR ultrasound[Title/Abstract]"
    ") AND humans[MeSH Terms] AND english[Language]"
)


def request_text(url: str, params: dict[str, Any], timeout: int = 60) -> str:
    """调用 NCBI E-utilities 并返回文本响应。"""
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "MedReason-Agent/0.1 research pipeline"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def esearch_pubmed_ids(
    query: str,
    max_records: int,
    email: str | None = None,
    api_key: str | None = None,
) -> list[str]:
    """用 PubMed esearch 获取 PMID 列表。"""
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_records,
        "sort": "relevance",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    payload = json.loads(request_text(NCBI_ESEARCH_URL, params))
    return [str(pmid) for pmid in payload.get("esearchresult", {}).get("idlist", [])]


def text_content(element: ET.Element | None) -> str:
    """抽取 XML element 内所有文本并压缩空白。"""
    if element is None:
        return ""
    return " ".join(" ".join(element.itertext()).split())


def parse_pubmed_articles(xml_text: str) -> list[dict[str, str]]:
    """从 PubMed efetch XML 中解析标题、摘要、期刊和年份。"""
    root = ET.fromstring(xml_text)
    records: list[dict[str, str]] = []
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue

        pmid = text_content(medline.find("PMID"))
        article_node = medline.find("Article")
        if not pmid or article_node is None:
            continue

        title = text_content(article_node.find("ArticleTitle"))
        abstract = text_content(article_node.find("Abstract"))
        if not abstract:
            continue

        journal = text_content(article_node.find("Journal/Title"))
        year = (
            text_content(article_node.find("Journal/JournalIssue/PubDate/Year"))
            or text_content(article_node.find("Journal/JournalIssue/PubDate/MedlineDate"))
        )
        records.append(
            {
                "pmid": pmid,
                "doc_id": f"PMID{pmid}",
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "source": "pubmed",
            }
        )
    return records


def fetch_pubmed_articles(
    pmids: list[str],
    batch_size: int,
    email: str | None = None,
    api_key: str | None = None,
    sleep_seconds: float = 0.34,
) -> list[dict[str, str]]:
    """分批下载 PubMed article XML 并解析成 JSONL records。"""
    records: list[dict[str, str]] = []
    for start in tqdm(range(0, len(pmids), batch_size), desc="fetch", ascii=True):
        batch_pmids = pmids[start : start + batch_size]
        params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(batch_pmids),
            "retmode": "xml",
        }
        if email:
            params["email"] = email
        if api_key:
            params["api_key"] = api_key

        xml_text = request_text(NCBI_EFETCH_URL, params, timeout=120)
        records.extend(parse_pubmed_articles(xml_text))
        time.sleep(sleep_seconds)
    return records


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    """写出 PubMed 摘要 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """解析下载参数。"""
    parser = argparse.ArgumentParser(description="下载 PubMed 摘要用于 Phase 3 RAG。")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--email", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.34)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/Raw/pmc/pmc_abstracts.jsonl"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_records <= 0:
        raise ValueError("--max-records must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    pmids = esearch_pubmed_ids(
        query=args.query,
        max_records=args.max_records,
        email=args.email,
        api_key=args.api_key,
    )
    records = fetch_pubmed_articles(
        pmids=pmids,
        batch_size=args.batch_size,
        email=args.email,
        api_key=args.api_key,
        sleep_seconds=args.sleep_seconds,
    )
    output_path = resolve_project_path(args.output)
    write_jsonl(output_path, records)

    print(
        json.dumps(
            {
                "query": args.query,
                "requested_records": args.max_records,
                "num_pmids": len(pmids),
                "num_records_with_abstract": len(records),
                "output": str(output_path.relative_to(resolve_project_path("."))),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


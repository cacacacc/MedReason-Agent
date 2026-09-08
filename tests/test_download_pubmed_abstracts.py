from scripts.download_pubmed_abstracts import parse_pubmed_articles


def test_parse_pubmed_articles_keeps_records_with_abstract() -> None:
    xml_text = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>123</PMID>
      <Article>
        <ArticleTitle>Chest radiograph opacity</ArticleTitle>
        <Abstract>
          <AbstractText>Opacity may be visible on chest radiographs.</AbstractText>
        </Abstract>
        <Journal>
          <Title>Test Journal</Title>
          <JournalIssue>
            <PubDate><Year>2024</Year></PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>456</PMID>
      <Article>
        <ArticleTitle>No abstract record</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

    records = parse_pubmed_articles(xml_text)

    assert records == [
        {
            "pmid": "123",
            "doc_id": "PMID123",
            "title": "Chest radiograph opacity",
            "abstract": "Opacity may be visible on chest radiographs.",
            "journal": "Test Journal",
            "year": "2024",
            "source": "pubmed",
        }
    ]


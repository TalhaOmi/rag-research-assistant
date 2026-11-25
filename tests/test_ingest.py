import pymupdf

from paper_assistant.ingest import (
    Page,
    build_chunks,
    clean_text,
    load_pdf_pages,
    pages_to_chunks,
    strip_references,
    tokens_to_words,
)


def make_pdf(path, page_texts):
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    doc.save(path)
    doc.close()


def test_clean_text_joins_hyphenated_breaks_and_collapses_whitespace():
    raw = "inter-\npretable   models\n\nare  good"
    assert clean_text(raw) == "interpretable models are good"


def test_clean_text_normalises_ligatures():
    assert clean_text("eﬃcient ﬁne") == "efficient fine"


def test_clean_text_keeps_real_hyphens_with_capitals_or_digits():
    assert clean_text("state-of-the-art GPT-\n4") == "state-of-the-art GPT- 4"


def test_page_numbers_are_one_based(tmp_path):
    pdf = tmp_path / "2001.00001v1.pdf"
    make_pdf(pdf, ["first page text", "second page text", "third page text"])
    pages = load_pdf_pages(pdf)
    assert [p.page for p in pages] == [1, 2, 3]
    assert pages[0].paper_id == "2001.00001v1"
    assert "second page" in pages[1].text


def test_chunks_carry_paper_and_page_and_never_cross_pages():
    pages = [
        Page("p1", 1, " ".join(f"a{i}" for i in range(50))),
        Page("p1", 2, " ".join(f"b{i}" for i in range(50))),
    ]
    chunks = pages_to_chunks(pages, size_words=30, overlap_words=5, min_words=1)
    assert {c.page for c in chunks} == {1, 2}
    for chunk in chunks:
        prefix = "a" if chunk.page == 1 else "b"
        assert all(w.startswith(prefix) for w in chunk.text.split())
    assert chunks[0].chunk_id == "p1:p1:c0"


def test_tiny_chunks_are_dropped():
    pages = [Page("p", 1, "too short")]
    assert pages_to_chunks(pages, 30, 5, min_words=20) == []


def test_strip_references_cuts_bibliography():
    body = "Our method works well. " * 20
    refs = "References 1. Smith J (2019) A. 2. Lee K (2020) B. 3. Wu T (2018) C. 4. Roy P (2021) D. 5. Ng A (2017) E."
    pages = [Page("p", 1, body), Page("p", 2, body), Page("p", 3, body + refs), Page("p", 4, refs)]
    kept = strip_references(pages)
    assert [p.page for p in kept] == [1, 2, 3]
    assert "References" not in kept[-1].text
    assert "Smith" not in " ".join(p.text for p in kept)


def test_strip_references_leaves_papers_without_bibliography_alone():
    pages = [Page("p", i, "just body text here " * 10) for i in range(1, 6)]
    assert strip_references(pages) == pages


def test_tokens_to_words():
    assert tokens_to_words(500) == 375
    assert tokens_to_words(80) == 60


def test_build_chunks_from_pdf_dir(tmp_path):
    words = " ".join(f"word{i}" for i in range(120))
    make_pdf(tmp_path / "2001.00001v1.pdf", [words, words])
    chunks = build_chunks(tmp_path, chunk_tokens=80, overlap_tokens=20)
    assert chunks
    assert {c.paper_id for c in chunks} == {"2001.00001v1"}
    assert {c.page for c in chunks} == {1, 2}

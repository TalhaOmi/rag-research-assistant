from paper_assistant.generate import Citation, parse_inline_citations, split_citations

from .conftest import make_hit


def test_parses_single_citation():
    text = "SHAP explains predictions [2304.01543v1, p.4]."
    assert parse_inline_citations(text) == [Citation("2304.01543v1", 4)]


def test_parses_multiple_and_dedupes_in_order():
    text = "A [2304.01543v1, p.4]. B [1812.04608v2, p. 7]. C [2304.01543v1, p.4]."
    assert parse_inline_citations(text) == [
        Citation("2304.01543v1", 4),
        Citation("1812.04608v2", 7),
    ]


def test_parses_page_word_and_spacing_variants():
    assert parse_inline_citations("[ 2304.01543v1 , page 12 ]") == [Citation("2304.01543v1", 12)]


def test_parses_old_style_arxiv_ids():
    assert parse_inline_citations("[cs/9308101v1, p.2]") == [Citation("cs/9308101v1", 2)]


def test_ignores_non_citations():
    assert parse_inline_citations("See [1] and [Smith, 2020] and [p.4].") == []


def test_split_into_valid_and_invalid():
    hits = [make_hit("2304.01543v1", 4)]
    valid, invalid = split_citations(
        [Citation("2304.01543v1", 4), Citation("2304.01543v1", 5), Citation("9999.99999v1", 1)],
        hits,
    )
    assert valid == [Citation("2304.01543v1", 4)]
    assert invalid == [Citation("2304.01543v1", 5), Citation("9999.99999v1", 1)]

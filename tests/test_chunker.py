import pytest

from paper_assistant.ingest import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("one two three", size=10, overlap=2) == ["one two three"]


def test_chunks_respect_size():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, size=30, overlap=5)
    assert all(len(c.split()) <= 30 for c in chunks)


def test_chunks_overlap():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, size=30, overlap=5)
    assert chunks[0].split()[-5:] == chunks[1].split()[:5]


def test_no_words_lost():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, size=30, overlap=5)
    assert chunks[0].split()[0] == "0"
    assert chunks[-1].split()[-1] == "99"


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", size=10, overlap=2) == []


def test_exact_multiple_has_no_overlap_only_tail():
    text = " ".join(str(i) for i in range(30))
    chunks = chunk_text(text, size=30, overlap=5)
    assert len(chunks) == 1


def test_every_word_appears_in_some_chunk():
    words = [str(i) for i in range(257)]
    chunks = chunk_text(" ".join(words), size=40, overlap=10)
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(words)


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, 10), (10, -1), (5, 9)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", size=size, overlap=overlap)

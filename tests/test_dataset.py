import json
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "evals" / "dataset.jsonl"


def load():
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]


def test_dataset_is_well_formed():
    rows = load()
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"
    for r in rows:
        assert "?" in r["question"]
        if r["answerable"]:
            assert set(r["gold"]) == {"paper_id", "page", "evidence"}
            assert len(r["gold"]["evidence"].split()) >= 3
        else:
            assert r["gold"] is None


def test_dataset_has_both_kinds_of_questions():
    rows = load()
    assert sum(r["answerable"] for r in rows) >= 20
    assert sum(not r["answerable"] for r in rows) >= 5


def test_readme_results_block_is_replaced():
    import pytest

    from evals.run_eval import replace_results_block

    text = "intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro"
    new = replace_results_block(text, "| a | b |")
    assert (
        "old" not in new
        and "| a | b |" in new
        and new.startswith("intro")
        and new.endswith("outro")
    )
    with pytest.raises(ValueError):
        replace_results_block("no markers here", "x")

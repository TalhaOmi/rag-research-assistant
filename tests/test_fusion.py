from paper_assistant.retrieve import rrf_fuse


def test_item_ranked_well_by_both_lists_wins():
    fused = rrf_fuse([["a", "b", "c"], ["a", "b", "d"]])
    ids = [i for i, _ in fused]
    assert ids[:2] == ["a", "b"]
    assert set(ids[2:]) == {"c", "d"}


def test_agreement_beats_single_high_rank():
    # "x" is #1 in one list only; "y" is #2 in both lists
    fused = dict(rrf_fuse([["x", "y"], ["z", "y"]]))
    assert fused["y"] > fused["x"]


def test_score_formula():
    fused = dict(rrf_fuse([["a"], ["a"]], k=60))
    assert fused["a"] == 2 / 61


def test_items_from_only_one_list_are_kept():
    ids = [i for i, _ in rrf_fuse([["a"], ["b"]])]
    assert set(ids) == {"a", "b"}


def test_empty_rankings():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_scores_are_sorted_descending():
    fused = rrf_fuse([["a", "b", "c", "d"], ["d", "c", "b", "a"]])
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)

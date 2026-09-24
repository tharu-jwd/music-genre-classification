from concept_fusion.run_schema import seeds_for


def test_seed_policy():
    assert seeds_for("B1") == (0, 1, 2)
    assert seeds_for("F-Concat") == (0, 1, 2)
    assert seeds_for("F-Gated") == (0, 1, 2)
    assert seeds_for("C-I") == (0,)
    assert seeds_for("F-Shortcut") == (0,)
    assert seeds_for("F-Hidden") == (0,)

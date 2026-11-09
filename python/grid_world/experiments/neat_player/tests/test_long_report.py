"""
Tests for experiments/neat_player/long_report.py.

period_ends(results)                           running total of len(val_scores)
curve_text(results)                            gens, top, sample, gap per period
compare_text(base, other, base_name, other_name)   two runs side by side, other - base
summary_text(base_by_seed, other_by_seed, generation, other_name, base_name)

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

from experiments.neat_player.long_report import period_ends, curve_text, compare_text, summary_text

###############################################################################
# helpers
###############################################################################

def period(n, top, sample):
    return {"val_scores": [0.0] * n, "top_target": top, "sample_target": sample}


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False

###############################################################################
# tests
###############################################################################

def test_period_ends_is_a_running_total():
    assert period_ends([period(30, 0, 0), period(30, 0, 0), period(10, 0, 0)]) == [30, 60, 70]
    assert period_ends([]) == []


def test_curve_text_lines():
    lines = curve_text([period(30, 0.25, 0.1), period(30, 0.5, 0.375)]).split("\n")
    assert lines[0].split() == ["gens", "top", "sample", "gap"]
    assert lines[2].split() == ["30", "0.250", "0.100", "0.150"]
    assert lines[3].split() == ["60", "0.500", "0.375", "0.125"]
    assert len(lines) == 4


def test_curve_text_empty_raises():
    assert raises(ValueError, curve_text, [])


def test_compare_text_differences_are_other_minus_base():
    base = [period(30, 0.25, 0.125), period(30, 0.5, 0.25)]
    other = [period(30, 0.375, 0.125), period(30, 0.25, 0.5)]
    lines = compare_text(base, other, "one", "many").split("\n")
    assert lines[2].split() == ["30", "0.250", "0.125", "0.375", "0.125", "+0.125", "+0.000"]
    assert lines[3].split() == ["60", "0.500", "0.250", "0.250", "0.500", "-0.250", "+0.250"]


def test_compare_text_names_are_in_the_header():
    head = compare_text([period(30, 0, 0)], [period(30, 0, 0)], "one", "many").split("\n")[0]
    assert "one top" in head and "many top" in head


def test_compare_text_different_periods_raise():
    assert raises(ValueError, compare_text, [period(30, 0, 0)], [period(20, 0, 0)], "a", "b")
    assert raises(ValueError, compare_text, [period(30, 0, 0)], [period(30, 0, 0), period(30, 0, 0)], "a", "b")
    assert raises(ValueError, compare_text, [], [], "a", "b")


def test_summary_mean_positives_and_range():
    base = {1: [period(30, 0.5, 0.25)], 2: [period(30, 0.25, 0.25)], 3: [period(30, 0.5, 0.25)]}
    other = {1: [period(30, 0.25, 0.5)], 2: [period(30, 0.5, 0.25)], 3: [period(30, 0.75, 0.5)]}
    # top differences: -0.25, +0.25, +0.25 ; sample: +0.25, 0.0, +0.25
    text = summary_text(base, other, 30, "many", "one")
    assert "seeds [1, 2, 3]" in text and "many minus one" in text
    assert "top +0.083 (positive on 2 of 3, from -0.250 to +0.250)" in text
    assert "sample +0.167 (positive on 2 of 3, from +0.000 to +0.250)" in text


def test_summary_uses_only_seeds_present_in_both():
    base = {1: [period(30, 0.5, 0.5)], 2: [period(30, 0.0, 0.0)]}
    other = {2: [period(30, 0.5, 0.5)], 3: [period(30, 0.0, 0.0)]}
    text = summary_text(base, other, 30, "b", "a")
    assert "seeds [2]" in text and "top +0.500 (positive on 1 of 1" in text


def test_summary_picks_the_period_that_ends_at_the_generation():
    base = {1: [period(30, 0.0, 0.0), period(30, 0.5, 0.5)]}
    other = {1: [period(30, 0.25, 0.25), period(30, 0.5, 0.5)]}
    assert "top +0.250" in summary_text(base, other, 30, "b", "a")
    assert "top +0.000" in summary_text(base, other, 60, "b", "a")


def test_summary_errors():
    base = {1: [period(30, 0, 0)]}
    assert raises(ValueError, summary_text, base, {2: [period(30, 0, 0)]}, 30, "b", "a")   # no common seed
    assert raises(ValueError, summary_text, base, base, 45, "b", "a")                      # no period ends there
    two = {1: [period(30, 0, 0), period(30, 0, 0)]}
    assert raises(ValueError, summary_text, two, two, 45, "b", "a")                        # between two period ends

###############################################################################
# run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"long_report: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()
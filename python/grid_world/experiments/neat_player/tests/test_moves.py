"""
Tests for experiments/neat_player/moves.py: move_stats.

move_stats(positions) -> {"steps", "distinct", "stay", "flip"}

    - steps    : len(positions)
    - distinct : number of different cells
    - stay     : positions[i] == positions[i-1]   (the move did nothing)
    - flip     : positions[i] == positions[i-2] and positions[i] != positions[i-1]   (A, B, A)
    - A stay is never also counted as a flip.
    - An empty list gives zeros; the input is not changed.

Run with pytest, or run this file directly to get a single pass/fail line.
"""

###############################################################################
# libraries
###############################################################################

from experiments.neat_player.moves import move_stats, behaviour_table

###############################################################################
# tests
###############################################################################

def test_empty_list_gives_zeros():
    assert move_stats([]) == {"steps": 0, "distinct": 0, "stay": 0, "flip": 0}


def test_single_position():
    assert move_stats([(3, 4)]) == {"steps": 1, "distinct": 1, "stay": 0, "flip": 0}


def test_straight_walk_has_no_stay_and_no_flip():
    result = move_stats([(0, 0), (0, 1), (0, 2), (0, 3)])
    assert result == {"steps": 4, "distinct": 4, "stay": 0, "flip": 0}


def test_standing_still_counts_as_stay():
    result = move_stats([(1, 1), (1, 1), (1, 1)])
    assert result == {"steps": 3, "distinct": 1, "stay": 2, "flip": 0}


def test_back_and_forth_counts_flips():
    # A B A B A : positions 2, 3 and 4 each return to the cell of two decisions before
    result = move_stats([(0, 0), (0, 1), (0, 0), (0, 1), (0, 0)])
    assert result == {"steps": 5, "distinct": 2, "stay": 0, "flip": 3}


def test_a_stay_is_not_a_flip():
    # A A A: the player never left A. Two stays, no flip.
    assert move_stats([(5, 5), (5, 5), (5, 5)])["flip"] == 0
    # A B B: second B is a stay, not a flip
    result = move_stats([(0, 0), (0, 1), (0, 1)])
    assert result["stay"] == 1 and result["flip"] == 0


def test_stay_then_return_is_not_a_flip():
    # A B B A: the last A is compared with positions[1] = B, so it is not a flip
    result = move_stats([(0, 0), (0, 1), (0, 1), (0, 0)])
    assert result == {"steps": 4, "distinct": 2, "stay": 1, "flip": 0}


def test_a_loop_of_four_cells_is_not_a_flip():
    square = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0), (0, 1)]
    result = move_stats(square)
    assert result == {"steps": 6, "distinct": 4, "stay": 0, "flip": 0}


def test_mixed_game():
    # A B A A C : flip at index 2, stay at index 3, nothing at index 4
    a, b, c = (0, 0), (0, 1), (0, 2)
    result = move_stats([a, b, a, a, c])
    assert result == {"steps": 5, "distinct": 3, "stay": 1, "flip": 1}


def test_input_is_not_changed():
    positions = [(0, 0), (0, 1), (0, 0)]
    before = list(positions)
    move_stats(positions)
    assert positions == before


def test_accepts_a_tuple_of_positions():
    result = move_stats(((0, 0), (0, 1), (0, 0)))
    assert result["flip"] == 1 and result["steps"] == 3

def row(seed, outcome, steps, distinct, stay, flip):
    return {"seed": seed, "outcome": outcome, "steps": steps, "distinct": distinct, "stay": stay, "flip": flip}


def test_table_has_a_line_per_game_and_one_summary_per_outcome():
    rows = [row(10, "GoalReached", 20, 20, 0, 0), row(11, "Timeout", 100, 5, 60, 30), row(12, "GoalReached", 30, 28, 2, 0)]
    lines = behaviour_table(rows).split("\n")
    assert lines[0].split() == ["seed", "outcome", "steps", "distinct", "stay", "flip"]
    assert lines[2].split() == ["10", "GoalReached", "20", "20", "0", "0"]
    assert lines[3].split() == ["11", "Timeout", "100", "5", "60", "30"]
    assert lines[4].split() == ["12", "GoalReached", "30", "28", "2", "0"]
    assert lines[5] == ""
    assert len(lines) == 8                       # header, rule, 3 games, blank, 2 summaries


def test_table_summary_shares_are_over_all_steps_of_the_outcome():
    rows = [row(1, "Timeout", 100, 5, 60, 30), row(2, "Timeout", 100, 9, 20, 10), row(3, "GoalReached", 10, 10, 1, 0)]
    lines = behaviour_table(rows).split("\n")
    timeout = [l for l in lines if l.startswith("Timeout")][0]
    goal = [l for l in lines if l.startswith("GoalReached")][0]
    assert timeout.split()[1] == "2" and "stay  40.0%" in timeout and "flip  20.0%" in timeout
    assert goal.split()[1] == "1" and "stay  10.0%" in goal and "flip   0.0%" in goal


def test_table_outcomes_are_in_order_of_first_appearance():
    rows = [row(1, "Caught", 10, 10, 0, 0), row(2, "GoalReached", 10, 10, 0, 0), row(3, "Caught", 10, 10, 0, 0)]
    summaries = behaviour_table(rows).split("\n")[-2:]
    assert summaries[0].startswith("Caught") and summaries[1].startswith("GoalReached")


def test_table_with_zero_steps_does_not_divide_by_zero():
    text = behaviour_table([row(1, "Timeout", 0, 0, 0, 0)])
    assert "stay   0.0% of steps" in text


def test_table_empty_raises():
    try:
        behaviour_table([])
    except ValueError:
        return
    raise AssertionError("expected ValueError")

###############################################################################
# run
###############################################################################

def run():
    here = run.__code__.co_filename
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and f.__code__.co_filename == here]
    for name, fn in tests:
        fn()
    print(f"moves: all {len(tests)} checks passed")


if __name__ == "__main__":
    run()
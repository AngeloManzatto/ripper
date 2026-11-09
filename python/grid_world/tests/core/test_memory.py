"""
Created on Wed Oct 14 16:02:40 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# libraries
###############################################################################

from core.memory import recent_visit_flags


def test_no_history_gives_no_flags():
    assert recent_visit_flags((5, 5), []) == [0.0, 0.0, 0.0, 0.0]


def test_each_direction_lights_its_own_flag():
    origin = (5, 5)
    assert recent_visit_flags(origin, [(4, 5)]) == [1.0, 0.0, 0.0, 0.0]   # up
    assert recent_visit_flags(origin, [(6, 5)]) == [0.0, 1.0, 0.0, 0.0]   # down
    assert recent_visit_flags(origin, [(5, 4)]) == [0.0, 0.0, 1.0, 0.0]   # left
    assert recent_visit_flags(origin, [(5, 6)]) == [0.0, 0.0, 0.0, 1.0]   # right


def test_current_cell_and_far_cells_light_nothing():
    assert recent_visit_flags((5, 5), [(5, 5), (2, 2), (7, 5), (6, 6)]) == [0.0] * 4


def test_several_visited_neighbors_and_repeats():
    assert recent_visit_flags((5, 5), [(4, 5), (5, 6), (4, 5)]) == [1.0, 0.0, 0.0, 1.0]


def run():
    test_no_history_gives_no_flags()
    test_each_direction_lights_its_own_flag()
    test_current_cell_and_far_cells_light_nothing()
    test_several_visited_neighbors_and_repeats()
    print("memory features: all checks passed")


if __name__ == "__main__":
    run()
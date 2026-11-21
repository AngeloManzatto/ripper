//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::grid_world::core::grid::Grid;

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn a_new_grid_has_its_size_and_is_filled() {
    let grid = Grid::new(4, 3, 7);

    assert_eq!(grid.width(), 4);
    assert_eq!(grid.height(), 3);
    for row in 0..3 {
        for col in 0..4 {
            assert_eq!(grid.get((row, col)), Some(&7));
        }
    }
}

#[test]
fn a_grid_can_hold_anything_that_can_be_cloned() {
    let letters = Grid::new(2, 2, 'x');
    let names = Grid::new(1, 1, String::from("wall"));

    assert_eq!(letters.get((1, 1)), Some(&'x'));
    assert_eq!(names.get((0, 0)), Some(&String::from("wall")));
}

#[test]
fn positions_are_row_then_column() {
    let grid = Grid::new(4, 3, 0);

    assert!(grid.in_bounds((2, 3)));
    assert!(!grid.in_bounds((3, 0)));
    assert!(!grid.in_bounds((0, 4)));
}

#[test]
fn outside_the_grid_there_is_nothing() {
    let grid = Grid::new(4, 3, 0);

    assert_eq!(grid.get((3, 0)), None);
    assert_eq!(grid.get((0, 4)), None);
}

#[test]
fn a_column_past_the_edge_does_not_wrap_to_the_next_row() {
    let grid = Grid::new(4, 3, 0);

    assert_eq!(grid.get((0, 4)), None);
}

#[test]
fn an_empty_grid_has_no_cells() {
    let grid = Grid::new(0, 0, 0);

    assert_eq!(grid.get((0, 0)), None);
}

#[test]
fn grids_with_the_same_size_and_cells_are_equal() {
    assert_eq!(Grid::new(2, 2, 1), Grid::new(2, 2, 1));
    assert_ne!(Grid::new(2, 2, 1), Grid::new(2, 2, 2));
    assert_ne!(Grid::new(2, 1, 1), Grid::new(1, 2, 1));
}

#[test]
fn a_cell_can_be_changed_through_get_mut() {
    let mut grid = Grid::new(4, 3, 0);

    *grid.get_mut((1, 2)).unwrap() = 9;

    assert_eq!(grid.get((1, 2)), Some(&9));
}

#[test]
fn changing_one_cell_leaves_the_others_alone() {
    let mut grid = Grid::new(4, 3, 0);

    *grid.get_mut((1, 2)).unwrap() = 9;

    let mut changed = 0;
    for row in 0..3 {
        for col in 0..4 {
            if grid.get((row, col)) != Some(&0) {
                changed += 1;
            }
        }
    }
    assert_eq!(changed, 1);
}

#[test]
fn outside_the_grid_there_is_no_cell_to_change() {
    let mut grid = Grid::new(4, 3, 0);

    assert_eq!(grid.get_mut((3, 0)), None);
    assert_eq!(grid.get_mut((0, 4)), None);
    assert_eq!(grid, Grid::new(4, 3, 0));
}

#[test]
fn the_last_cell_of_the_last_row_can_be_changed() {
    let mut grid = Grid::new(4, 3, 0);

    *grid.get_mut((2, 3)).unwrap() = 5;

    assert_eq!(grid.get((2, 3)), Some(&5));
    assert_eq!(grid.get((2, 2)), Some(&0));
}
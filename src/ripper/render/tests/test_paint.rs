//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::sync::Arc;

use super::super::frame::{blank_frame, Cell};
use super::super::paint::{paint_terrain};

use crate::ripper::content::store::{Content, TerrainId};
use crate::ripper::core::color::Color;
use crate::ripper::core::grid::Grid;
use crate::ripper::test_support::{base_content, empty, void, wall, EMPTY_ID, WALL_ID};

use crate::ripper::render::paint::paint_level;
use crate::ripper::render::terminal::frame_to_text;
use crate::ripper::world::level::Level;

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn a_cell_takes_its_glyph_and_both_colors_from_its_terrain() {
    let content = base_content();
    let grid = Grid::new(1, 1, WALL_ID);
    let mut frame = blank_frame(1, 1);

    paint_terrain(&mut frame, &grid, &content);

    let expected = Cell::new('#', Color::new(200, 200, 200), Color::new(48, 48, 48));
    assert_eq!(frame.get((0, 0)), Some(&expected));
}

#[test]
fn every_cell_follows_its_own_terrain() {
    let content = base_content();
    let mut grid = Grid::new(2, 2, EMPTY_ID);
    *grid.get_mut((0, 1)).unwrap() = WALL_ID;
    *grid.get_mut((1, 0)).unwrap() = WALL_ID;
    let mut frame = blank_frame(2, 2);

    paint_terrain(&mut frame, &grid, &content);

    assert_eq!(frame.get((0, 0)).unwrap().glyph, '.');
    assert_eq!(frame.get((0, 1)).unwrap().glyph, '#');
    assert_eq!(frame.get((1, 0)).unwrap().glyph, '#');
    assert_eq!(frame.get((1, 1)).unwrap().glyph, '.');
}

#[test]
fn a_terrain_without_a_glyph_is_a_space_that_keeps_its_colors() {
    let content = Content::new(vec![empty(), wall(), void()]);
    let grid = Grid::new(1, 1, TerrainId(2));
    let mut frame = blank_frame(1, 1);

    paint_terrain(&mut frame, &grid, &content);

    let cell = frame.get((0, 0)).unwrap();
    assert_eq!(cell.glyph, ' ');
    assert_eq!(cell.bg_color, void().bg_color);
}

#[test]
fn the_cells_follow_the_content_not_the_number() {
    let swapped = Content::new(vec![wall(), empty()]);
    let grid = Grid::new(1, 1, TerrainId(0));
    let mut frame = blank_frame(1, 1);

    paint_terrain(&mut frame, &grid, &swapped);

    assert_eq!(frame.get((0, 0)).unwrap().glyph, '#');
}

#[test]
fn drawing_replaces_what_was_in_the_frame() {
    let content = base_content();
    let grid = Grid::new(1, 1, EMPTY_ID);
    let mut frame = blank_frame(1, 1);
    *frame.get_mut((0, 0)).unwrap() = Cell::new('X', Color::new(1, 2, 3), Color::new(4, 5, 6));

    paint_terrain(&mut frame, &grid, &content);

    assert_eq!(frame.get((0, 0)).unwrap().glyph, '.');
}

#[test]
fn cells_beyond_the_terrain_are_left_alone() {
    let content = base_content();
    let grid = Grid::new(2, 2, WALL_ID);
    let mut frame = blank_frame(3, 3);

    paint_terrain(&mut frame, &grid, &content);

    assert_eq!(frame.get((1, 1)).unwrap().glyph, '#');
    assert_eq!(frame.get((2, 2)), Some(&Cell::blank()));
    assert_eq!(frame.get((0, 2)), Some(&Cell::blank()));
}

#[test]
#[should_panic(expected = "at least as large")]
fn a_frame_too_small_for_the_terrain_stops_the_drawing() {
    let content = base_content();
    let grid = Grid::new(3, 3, WALL_ID);
    let mut frame = blank_frame(2, 2);

    paint_terrain(&mut frame, &grid, &content);
}

#[test]
fn the_painted_level_has_the_size_of_its_terrain() {
    let level = Level::from_layout("#.#\n...", Arc::new(base_content())).unwrap();
    let frame = paint_level(&level);
    assert_eq!(frame.width(), 3);
    assert_eq!(frame.height(), 2);
}

#[test]
fn a_painted_level_reads_back_as_the_layout_it_came_from() {
    let layout = "#####\n#...#\n#####";
    let level = Level::from_layout(layout, Arc::new(base_content())).unwrap();
    assert_eq!(frame_to_text(&paint_level(&level)), layout);
}
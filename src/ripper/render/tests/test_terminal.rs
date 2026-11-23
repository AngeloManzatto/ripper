//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::super::frame::blank_frame;

use super::super::terminal::{frame_to_text, frame_to_ansi};
use super::super::frame::Cell;

use crate::ripper::content::store::{Content, TerrainId};
use crate::ripper::core::grid::Grid;
use crate::ripper::world::layout::parse_layout;
use crate::ripper::render::paint::paint_terrain;
use crate::ripper::test_support::{
    base_content, block, empty, void, wall
};
use crate::ripper::core::color::Color;

//-----------------------------------------------------
// Helpers
//-----------------------------------------------------

fn show(grid: &Grid<TerrainId>, content: &Content) -> String {
    let mut frame = blank_frame(grid.width(), grid.height());
    paint_terrain(&mut frame, grid, content);
    frame_to_text(&frame)
}

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn a_grid_is_drawn_with_the_glyph_of_each_terrain() {

    let content = base_content();

    let mut grid = Grid::new(2, 2, TerrainId(0));
    *grid.get_mut((0, 0)).unwrap() = TerrainId(1);

    assert_eq!(show(&grid, &content), "#.\n..");
}

#[test]
fn a_layout_drawn_again_gives_the_same_text() {
    let content = base_content();

    for text in ["#", "###\n#.#\n###", ".#.\n#.#", "....\n....\n####"] {
        let grid = parse_layout(text, &content).unwrap();

        assert_eq!(show(&grid, &content), text);
    }
}

#[test]
fn the_drawing_has_no_final_newline() {

    let content = base_content();

    let grid = parse_layout("##\n..\n", &content).unwrap();

    assert_eq!(show(&grid, &content), "##\n..");
}

#[test]
fn a_terrain_without_a_glyph_is_drawn_as_a_space() {
    let content = Content::new(vec![empty(), wall(), void()]);
    let grid = Grid::new(2, 1, TerrainId(2));

    assert_eq!(show(&grid, &content), "  ");
}

#[test]
fn a_glyph_wider_than_one_byte_is_drawn_whole() {
    let content = Content::new(vec![empty(), wall(), block()]);
    let grid = parse_layout("▓▓\n▓.", &content).unwrap();

    assert_eq!(show(&grid, &content), "▓▓\n▓.");
}

#[test]
fn the_drawing_follows_the_content_not_the_number() {
    let swapped = Content::new(vec![wall(), empty()]);
    let grid = Grid::new(1, 1, TerrainId(0));

    assert_eq!(show(&grid, &swapped), "#");
}

#[test]
fn look_at_it() {
    let content = Content::new(vec![empty(), wall()]);
    let layout = "##########\n#........#\n#..####..#\n#..#..#..#\n#........#\n##########";
    let grid = parse_layout(layout, &content).unwrap();

    println!("\n{}\n", show(&grid, &content));
}

#[test]
fn a_blank_frame_becomes_rows_of_spaces() {
    let frame = blank_frame(3, 2);
    assert_eq!(frame_to_text(&frame), "   \n   ");
}

#[test]
fn a_single_cell_frame_has_no_newline_at_all() {
    let frame = blank_frame(1, 1);
    assert_eq!(frame_to_text(&frame), " ");
}

#[test]
fn there_is_no_newline_after_the_last_row() {
    let text = frame_to_text(&blank_frame(2, 3));
    assert!(!text.ends_with('\n'));
}

#[test]
fn the_text_has_one_line_per_frame_row() {
    let text = frame_to_text(&blank_frame(4, 5));
    assert_eq!(text.lines().count(), 5);
}

#[test]
fn one_cell_is_wrapped_in_its_colours_and_a_reset() {
    let mut frame = blank_frame(1, 1);
    *frame.get_mut((0, 0)).unwrap() = Cell::new('#', Color::new(1, 2, 3), Color::new(4, 5, 6));

    assert_eq!(
        frame_to_ansi(&frame),
        "\x1b[38;2;1;2;3m\x1b[48;2;4;5;6m#\x1b[0m"
    );
}

#[test]
fn a_blank_cell_is_white_on_black() {
    assert_eq!(
        frame_to_ansi(&blank_frame(1, 1)),
        "\x1b[38;2;255;255;255m\x1b[48;2;0;0;0m \x1b[0m"
    );
}

#[test]
fn every_cell_carries_its_own_colours() {
    let mut frame = blank_frame(2, 1);
    *frame.get_mut((0, 0)).unwrap() = Cell::new('a', Color::new(1, 1, 1), Color::new(2, 2, 2));
    *frame.get_mut((0, 1)).unwrap() = Cell::new('b', Color::new(3, 3, 3), Color::new(4, 4, 4));

    assert_eq!(
        frame_to_ansi(&frame),
        "\x1b[38;2;1;1;1m\x1b[48;2;2;2;2ma\x1b[38;2;3;3;3m\x1b[48;2;4;4;4mb\x1b[0m"
    );
}

#[test]
fn each_row_ends_with_a_reset_and_the_last_has_no_newline() {
    let frame = blank_frame(1, 2);
    let blank = "\x1b[38;2;255;255;255m\x1b[48;2;0;0;0m \x1b[0m";

    assert_eq!(frame_to_ansi(&frame), format!("{blank}\n{blank}"));
}
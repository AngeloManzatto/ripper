//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::super::frame::{blank_frame, Cell, Frame};

use crate::ripper::core::color::Color;

//-----------------------------------------------------
// Helpers
//-----------------------------------------------------
fn red() -> Color {
    Color::new(255, 0, 0)
}

fn blue() -> Color {
    Color::new(0, 0, 255)
}

#[test]
fn a_cell_keeps_its_glyph_and_both_colors() {
    let cell = Cell::new('#', red(), blue());

    assert_eq!(cell.glyph, '#');
    assert_eq!(cell.fg_color, red());
    assert_eq!(cell.bg_color, blue());
}

#[test]
fn a_cell_is_copied_not_moved() {
    let first = Cell::new('#', red(), blue());
    let second = first;

    assert_eq!(first, second);
}

#[test]
fn cells_differ_when_any_part_differs() {
    let base = Cell::new('#', red(), blue());

    assert_ne!(base, Cell::new('.', red(), blue()));
    assert_ne!(base, Cell::new('#', blue(), blue()));
    assert_ne!(base, Cell::new('#', red(), red()));
}

#[test]
fn a_blank_cell_is_a_space_on_black() {
    let blank = Cell::blank();

    assert_eq!(blank.glyph, ' ');
    assert_eq!(blank.bg_color, Color::new(0, 0, 0));
}

#[test]
fn a_blank_frame_has_the_requested_size() {
    let frame = blank_frame(4, 3);

    assert_eq!(frame.width(), 4);
    assert_eq!(frame.height(), 3);
}

#[test]
fn every_cell_of_a_blank_frame_is_blank() {
    let frame = blank_frame(3, 2);

    for row in 0..frame.height() {
        for col in 0..frame.width() {
            assert_eq!(frame.get((row, col)), Some(&Cell::blank()));
        }
    }
}

#[test]
fn a_cell_written_into_a_frame_is_read_back_and_its_neighbours_stay_blank() {
    let mut frame: Frame = blank_frame(3, 3);
    let painted = Cell::new('#', red(), blue());

    *frame.get_mut((1, 1)).unwrap() = painted;

    assert_eq!(frame.get((1, 1)), Some(&painted));
    assert_eq!(frame.get((0, 1)), Some(&Cell::blank()));
    assert_eq!(frame.get((1, 2)), Some(&Cell::blank()));
}

#[test]
fn a_position_outside_the_frame_cannot_be_painted() {
    let mut frame = blank_frame(2, 2);

    assert!(frame.get_mut((2, 0)).is_none());
}
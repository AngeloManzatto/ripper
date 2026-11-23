//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::super::frame::{blank_frame, Cell};
use super::super::pixels::frame_to_rgb;

use crate::ripper::core::color::Color;

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

fn cell_with_background(n: u8) -> Cell {
    Cell::new('x', Color::new(0, 0, 0), Color::new(n, n, n))
}

#[test]
fn a_blank_frame_is_black_pixels() {
    assert_eq!(frame_to_rgb(&blank_frame(2, 1)), vec![0, 0, 0, 0, 0, 0]);
}

#[test]
fn there_are_three_numbers_per_cell() {
    assert_eq!(frame_to_rgb(&blank_frame(4, 3)).len(), 4 * 3 * 3);
}

#[test]
fn the_pixel_is_the_background_not_the_foreground() {
    let mut frame = blank_frame(1, 1);
    *frame.get_mut((0, 0)).unwrap() = Cell::new('#', Color::new(1, 2, 3), Color::new(4, 5, 6));

    assert_eq!(frame_to_rgb(&frame), vec![4, 5, 6]);
}

#[test]
fn the_pixels_come_row_by_row() {
    let mut frame = blank_frame(2, 2);
    *frame.get_mut((0, 0)).unwrap() = cell_with_background(1);
    *frame.get_mut((0, 1)).unwrap() = cell_with_background(2);
    *frame.get_mut((1, 0)).unwrap() = cell_with_background(3);
    *frame.get_mut((1, 1)).unwrap() = cell_with_background(4);

    assert_eq!(
        frame_to_rgb(&frame),
        vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    );
}
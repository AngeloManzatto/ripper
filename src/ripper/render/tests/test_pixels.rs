//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::super::frame::{blank_frame, Cell};
use super::super::pixels::{frame_to_rgb, scale_rgb, pack_rgb};

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

#[test]
fn a_factor_of_one_changes_nothing() {
    let rgb = vec![1, 2, 3, 4, 5, 6];
    assert_eq!(scale_rgb(&rgb, 2, 1, 1), rgb);
}

#[test]
fn one_pixel_becomes_a_block() {
    assert_eq!(scale_rgb(&[1, 2, 3], 1, 1, 2), [1, 2, 3].repeat(4));
}

#[test]
fn a_wide_picture_repeats_each_pixel_across_and_each_row_down() {
    // two pixels side by side: colour 1 then colour 2
    let rgb = vec![1, 1, 1, 2, 2, 2];
    let row = vec![1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];

    assert_eq!(scale_rgb(&rgb, 2, 1, 2), [row.clone(), row].concat());
}

#[test]
fn a_tall_picture_keeps_its_rows_in_order() {
    // two pixels stacked: colour 1 above colour 2
    let rgb = vec![1, 1, 1, 2, 2, 2];
    let one = vec![1; 6];
    let two = vec![2; 6];

    assert_eq!(
        scale_rgb(&rgb, 1, 2, 2),
        [one.clone(), one, two.clone(), two].concat()
    );
}

#[test]
fn the_output_has_the_scaled_size() {
    let rgb = vec![0; 10 * 6 * 3];
    assert_eq!(scale_rgb(&rgb, 10, 6, 16).len(), 160 * 96 * 3);
}

#[test]
#[should_panic(expected = "the factor must be at least 1")]
fn a_factor_of_zero_is_refused() {
    scale_rgb(&[1, 2, 3], 1, 1, 0);
}

#[test]
#[should_panic(expected = "do not match the width and height")]
fn pixels_that_do_not_fit_the_size_are_refused() {
    scale_rgb(&[1, 2, 3], 2, 1, 1);
}

#[test]
fn a_pixel_is_packed_as_red_green_blue() {
    assert_eq!(pack_rgb(&[1, 2, 3]), vec![0x010203]);
}

#[test]
fn each_channel_lands_in_its_own_byte() {
    assert_eq!(pack_rgb(&[255, 0, 0]), vec![0xFF0000]);
    assert_eq!(pack_rgb(&[0, 255, 0]), vec![0x00FF00]);
    assert_eq!(pack_rgb(&[0, 0, 255]), vec![0x0000FF]);
}

#[test]
fn there_is_one_number_per_pixel_in_order() {
    assert_eq!(pack_rgb(&[1, 1, 1, 2, 2, 2]), vec![0x010101, 0x020202]);
}

#[test]
fn no_pixels_give_no_numbers() {
    assert_eq!(pack_rgb(&[]), Vec::<u32>::new());
}

#[test]
#[should_panic(expected = "multiple of 3")]
fn bytes_that_are_not_whole_pixels_are_refused() {
    pack_rgb(&[1, 2, 3, 4]);
}
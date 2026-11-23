//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::ripper::render::frame::Frame;

//-----------------------------------------------------
// Frame to rgb
//-----------------------------------------------------

pub fn frame_to_rgb(frame: &Frame) -> Vec<u8> {
    let mut pixels = Vec::new();

    for row in 0..frame.height() {
        for col in 0..frame.width() {

            // 1. get the cell at (row, col)
            let cell = frame.get((row, col)).expect("cell in range must exist");

            // 2. push the red, green and blue of its background colour
            let bg_color = cell.bg_color;

            pixels.extend([bg_color.r, bg_color.g, bg_color.b]);
        }
    }

    pixels
}
//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::ripper::render::frame::Frame;

//-----------------------------------------------------
// Frame to RGB
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

//-----------------------------------------------------
// Scale RGB
//-----------------------------------------------------

pub fn scale_rgb(rgb: &[u8], width: usize, height: usize, factor: usize) -> Vec<u8> {
    assert!(factor > 0, "the factor must be at least 1");
    assert_eq!(
        rgb.len(),
        width * height * 3,
        "the pixels do not match the width and height"
    );

    let mut scaled = Vec::new();

    for y in 0..height * factor {
        for x in 0..width * factor {

            // 1. find where the first byte of the source pixel is
            let pos_x = x / factor;
            let pos_y = y / factor;

            let start = (pos_y * width + pos_x) * 3;

            // 2. push the three bytes: rgb[start], rgb[start + 1], rgb[start + 2]
            scaled.extend([rgb[start], rgb[start + 1], rgb[start + 2]]);
        }
    }

    scaled
}

//-----------------------------------------------------
// Pack RGB
//-----------------------------------------------------

pub fn pack_rgb(rgb: &[u8]) -> Vec<u32> {
    assert!(rgb.len() % 3 == 0, "the pixels must be a multiple of 3 bytes");

    let mut packed = Vec::new();

    for pixel in rgb.chunks(3) {

        // 1. widen the three bytes to u32
        let r = pixel[0] as u32;
        let g = pixel[1] as u32;
        let b = pixel[2] as u32;
        
        // 2. combine them: (r << 16) | (g << 8) | b
        let color = (r << 16) | (g << 8) | b;

        // 3. push the result
        packed.push(color);
        
    }

    packed
}
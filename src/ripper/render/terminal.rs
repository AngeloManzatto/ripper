//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::ripper::render::frame::Frame;

//-----------------------------------------------------
// Frame to text
//-----------------------------------------------------

pub fn frame_to_text(frame: &Frame) -> String {
    let mut text = String::new();

    for row in 0..frame.height() {
        for col in 0..frame.width() {
            // 1. get the cell at (col, row) from the frame
            let cell = frame.get((row, col)).expect("cell in range must exist");

            // 2. push its glyph onto `text`
            text.push(cell.glyph);
        }
        if row + 1 < frame.height() {
            // 3. push '\n', but only if this is not the last row
            text.push('\n');
        }
        
    }

    text
}

//-----------------------------------------------------
// Frame to ansi
//-----------------------------------------------------

pub fn frame_to_ansi(frame: &Frame) -> String {
    let mut text = String::new();

    for row in 0..frame.height() {
        for col in 0..frame.width() {
            // 1. get the cell
            let cell = frame.get((row, col)).expect("cell in range must exist");

            // 2. push_str the foreground code, then the background code
            let fg_color_str = format!("\x1b[38;2;{};{};{}m", 
                                               cell.fg_color.r, cell.fg_color.g, cell.fg_color.b);

            let bg_color_str = format!("\x1b[48;2;{};{};{}m", 
                                               cell.bg_color.r, cell.bg_color.g, cell.bg_color.b);

            // 3. push the glyph
            let glyph_str = format!("{}", cell.glyph);

            text.push_str(&fg_color_str);
            text.push_str(&bg_color_str);
            text.push_str(&glyph_str);

        }
        // 4. push_str the reset code
        text.push_str("\x1b[0m");

        // 5. push '\n' if this is not the last row
        if row + 1 < frame.height() {
            text.push('\n');
        }
    }

    text
}
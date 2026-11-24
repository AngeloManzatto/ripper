//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::path::PathBuf;
use std::process;
use std::sync::Arc;

use macroquad::prelude::*;

use ripper::ripper::packs::error::format_errors;
use ripper::ripper::packs::load_content;
use ripper::ripper::render::paint::paint_level;
use ripper::ripper::world::level::Level;

//-----------------------------------------------------
// Layout
//-----------------------------------------------------

const LAYOUT: &str = "\
##########
#........#
#..####..#
#..#..#..#
#........#
##########";

const CELL: f32 = 48.0;
const FONT_SIZE: u16 = 40;

//-----------------------------------------------------
// Main
//-----------------------------------------------------

#[macroquad::main("RIPPER")]
async fn main() {

    let packs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/ripper/assets/packs");

    let content = match load_content(&packs) {
        Ok(content) => Arc::new(content),
        Err(errors) => {
            eprintln!("{}", format_errors(&errors));
            process::exit(1);
        }
    };

    let level = match Level::from_layout(LAYOUT, content) {
        Ok(level) => level,
        Err(errors) => {
            eprintln!("{:?}", errors);
            process::exit(1);
        }
    };

    let frame = paint_level(&level);
    let (cols, rows) = (frame.width(), frame.height());

    request_new_screen_size(cols as f32 * CELL, rows as f32 * CELL);

    loop {
        clear_background(BLACK);

        for row in 0..rows {
            for col in 0..cols {
                let cell = frame.get((row, col)).expect("cell in range must exist");

                let x = col as f32 * CELL;
                let y = row as f32 * CELL;

                // the background: a filled square
                draw_rectangle(
                    x,
                    y,
                    CELL,
                    CELL,
                    Color::from_rgba(cell.bg_color.r, cell.bg_color.g, cell.bg_color.b, 255),
                );

                // the glyph, centred in the square
                if cell.glyph != ' ' {
                    let text = cell.glyph.to_string();
                    let size = measure_text(&text, None, FONT_SIZE, 1.0);

                    draw_text(
                        &text,
                        x + (CELL - size.width) / 2.0,
                        y + (CELL - size.height) / 2.0 + size.offset_y,
                        FONT_SIZE as f32,
                        Color::from_rgba(cell.fg_color.r, cell.fg_color.g, cell.fg_color.b, 255),
                    );
                }
            }
        }

        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        next_frame().await;
    }
}
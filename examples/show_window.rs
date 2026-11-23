//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::path::PathBuf;
use std::process;
use std::sync::Arc;

use ripper::ripper::packs::error::format_errors;
use ripper::ripper::packs::load_content;
use ripper::ripper::render::paint::paint_level;
use ripper::ripper::world::level::Level;

use ripper::ripper::render::pixels::{frame_to_rgb, scale_rgb, pack_rgb};

use minifb::{Key, Window, WindowOptions};

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

const SCALE: usize = 32;

fn main() {

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

    let frame =paint_level(&level);
    let width = level.terrain().width();
    let height = level.terrain().height();

    let rgb = frame_to_rgb(&frame);

    let scaled_rgb = scale_rgb(&rgb, width, height, SCALE);

    let buffer = pack_rgb(&scaled_rgb);

    // 2. the chain, using the signatures you wrote:
    //    - frame  = paint_level(&level)
    //    - width and height from the frame
    //    - rgb    = frame_to_rgb(&frame)
    //    - scaled = scale_rgb(&rgb, width, height, SCALE)
    //    - buffer = pack_rgb(&scaled)

    // 3. the window
    let (win_w, win_h) = (width * SCALE, height * SCALE);

    let mut window = Window::new("RIPPER", win_w, win_h, WindowOptions::default())
        .expect("could not open the window");
    window.set_target_fps(30);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window
            .update_with_buffer(&buffer, win_w, win_h)
            .expect("could not update the window");
    }
}
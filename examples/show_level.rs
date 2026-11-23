//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::path::PathBuf;
use std::process;
use std::sync::Arc;

use ripper::ripper::packs::error::format_errors;
use ripper::ripper::packs::load_content;
use ripper::ripper::render::terminal::frame_to_ansi;
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

//-----------------------------------------------------
// Main
//-----------------------------------------------------

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

    println!("{}", frame_to_ansi(&paint_level(&level)));
}
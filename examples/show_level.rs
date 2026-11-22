use std::path::PathBuf;
use std::process;
use std::sync::Arc;

use ripper::ripper::packs::error::format_errors;
use ripper::ripper::packs::load_content;
use ripper::ripper::render::terminal::draw_terrain;
use ripper::ripper::world::level::Level;

const LAYOUT: &str = "\
##########
#........#
#..####..#
#..#..#..#
#........#
##########";

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

    println!("{}", draw_terrain(level.terrain(), level.content()));
}
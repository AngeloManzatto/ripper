//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::ripper::content::store::{Content, TerrainId};
use crate::ripper::core::grid::Grid;
use crate::ripper::world::level::Level;

use super::frame::{Cell, Frame, blank_frame};

//-----------------------------------------------------
// Draw terrain
//-----------------------------------------------------

pub fn paint_terrain(frame: &mut Frame, grid: &Grid<TerrainId>, content: &Content) {
    assert!(
        frame.width() >= grid.width() && frame.height() >= grid.height(),
        "the frame must be at least as large as the terrain grid",
    );

    for row in 0..grid.height() {
        for col in 0..grid.width() {
            let id = grid.get((row, col)).expect("cell inside the grid");
            let terrain = content.terrain(*id);

            let glyph = terrain.glyph.unwrap_or(' ');
            let cell = Cell::new(glyph, terrain.fg_color, terrain.bg_color);

            *frame.get_mut((row, col)).expect("cell inside the frame") = cell;
        }
    }
}

//-----------------------------------------------------
// Paint level
//-----------------------------------------------------

pub fn paint_level(level: &Level) -> Frame {

    // 1. make a blank frame as big as the level's terrain grid
    let grid = level.terrain();
    let height = grid.height();
    let width = grid.width();

    let mut frame = blank_frame(width, height);
    let content = level.content();
    
    // 2. paint the terrain layer into it
    paint_terrain(&mut frame, grid, content);

    // 3. return the frame
    frame
}
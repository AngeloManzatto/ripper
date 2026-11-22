//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::ripper::content::store::{Content, TerrainId};
use crate::ripper::core::grid::Grid;

//-----------------------------------------------------
// Render text
//-----------------------------------------------------

pub fn draw_terrain(grid: &Grid<TerrainId>, content: &Content) -> String 
{

    let mut text = String::new();

    for row in 0..grid.height() {

        if row > 0 {
            text.push('\n');
        }

        for col in 0..grid.width() {
            let id = grid.get((row, col)).expect("cell inside the grid");
            let glyph = content.terrain(*id).glyph.unwrap_or(' ');
            text.push(glyph);
        }
    }

    text
}
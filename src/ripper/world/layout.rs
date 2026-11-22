//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::ripper::content::store::{Content, TerrainId};
use crate::ripper::core::grid::Grid;

//-----------------------------------------------------
// Layout Error
//-----------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum LayoutError {
    Empty,
    RaggedRow { row: usize, expected: usize, found: usize },
    UnknownGlyph { row: usize, col: usize, glyph: char },
}

//-----------------------------------------------------
// Parse layout
//-----------------------------------------------------

pub fn parse_layout(text: &str, content: &Content) -> Result<Grid<TerrainId>, Vec<LayoutError>> {

    // Extract lines into a vector of strings
    let rows: Vec<&str> = text.lines().collect();

    // Check for no empty grid definition
    if rows.is_empty() || rows[0].is_empty() {
        return Err(vec![LayoutError::Empty]);
    }

    let width = rows[0].chars().count();
    let mut grid = Grid::new(width, rows.len(), TerrainId(0));
    let mut errors = Vec::new();

    for (row, line) in rows.iter().enumerate() {
        let found = line.chars().count();
        if found != width {
            errors.push(LayoutError::RaggedRow { row, expected: width, found });
            continue;
        }
        for (col, glyph) in line.chars().enumerate() {
            match content.terrain_id_by_glyph(glyph) {
                Some(id) => {
                    if let Some(cell) = grid.get_mut((row, col)) {
                        *cell = id;
                    }
                }
                None => errors.push(LayoutError::UnknownGlyph { row, col, glyph }),
            }
        }
    }

    if errors.is_empty() {
        Ok(grid)
    } else {
        Err(errors)
    }
}
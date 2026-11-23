//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::ripper::core::color::Color;
use crate::ripper::core::grid::Grid;

//-----------------------------------------------------
// Cell
//-----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub glyph: char,
    pub fg_color: Color,
    pub bg_color: Color,
}

//-----------------------------------------------------
// Cell impl
//-----------------------------------------------------

impl Cell {
    pub fn new(glyph: char, fg_color: Color, bg_color: Color) -> Cell {
        Cell { glyph, fg_color, bg_color }
    }

    pub fn blank() -> Cell {
        Cell {
            glyph: ' ',
            fg_color: Color::new(255, 255, 255),
            bg_color: Color::new(0, 0, 0),
        }
    }
}

//-----------------------------------------------------
// Frame
//-----------------------------------------------------

pub type Frame = Grid<Cell>;

//-----------------------------------------------------
// Blank frame
//-----------------------------------------------------

pub fn blank_frame(width: usize, height: usize) -> Frame {
    Grid::new(width, height, Cell::blank())
}
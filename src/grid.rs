//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::fmt;

//-----------------------------------------------------
// Types
//-----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cell {
    Empty,
    Agent,
    Goal,
}

//-----------------------------------------------------
// Grid
//-----------------------------------------------------
pub struct Grid {
    pub width: usize,
    pub height: usize, 
    pub data: Vec<Vec<Cell>>,
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Cell::Empty => write!(f, "."),
            Cell::Agent => write!(f, "A"),
            Cell::Goal  => write!(f, "G"),
        }
    }
}

//-----------------------------------------------------
// Grid utilities
//-----------------------------------------------------

pub fn create_grid(width: usize, height: usize) -> Grid {
    let data = vec![vec![Cell::Empty; width]; height];
    Grid { width, height, data }
}

pub fn print_grid(grid: &Grid) {
    for row in &grid.data {
        for cell in row {
            print!("{} ", cell);
        }
        println!();
    }
}
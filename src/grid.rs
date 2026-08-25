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
    pub agent_start: (usize, usize),
    pub goal_pos: (usize, usize),
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

pub fn create_grid(
    width: usize, 
    height: usize, 
    agent_start: (usize, usize), 
    goal_pos: (usize, usize)) -> Grid {

        // Initialize the grid with Empty cells
        let data = vec![vec![Cell::Empty; width]; height];

        // Set the agent and goal positions
        Grid { width, height, data, agent_start, goal_pos }
}

//-----------------------------------------------------
// Reset Grid
//-----------------------------------------------------
pub fn reset_grid(grid: &mut Grid) {

    // Reset all cells to Empty
    for row in 0..grid.height {
        for col in 0..grid.width {
            grid.data[row][col] = Cell::Empty;
        }
    }

    // Set the agent and goal positions
    grid.data[grid.agent_start.0][grid.agent_start.1] = Cell::Agent;
    grid.data[grid.goal_pos.0][grid.goal_pos.1] = Cell::Goal;

}

//-----------------------------------------------------
// Print Grid
//-----------------------------------------------------
pub fn print_grid(grid: &Grid) {
    for row in &grid.data {
        for cell in row {
            print!("{} ", cell);
        }
        println!();
    }
}
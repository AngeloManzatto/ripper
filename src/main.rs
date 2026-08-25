//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::grid::Cell::Agent;

mod grid;

//-----------------------------------------------------
// Global Constants
//-----------------------------------------------------

const GRID_SIZE: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Up,
    Down,
    Left,
    Right,
    Reset,
    Exit,
}

#[derive(Debug, Clone, Copy)]
struct Position {
    row: usize,
    col: usize,
}

#[derive(Debug)]
struct StepResult {
    position: Position,
    reward: f64,
    done: bool
}

//-----------------------------------------------------
// Read input
//-----------------------------------------------------
fn read_input() -> Option<Action> {
    use std::io::{self, Write};

    print!("Enter action (w/a/s/d/r/q): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let input = input.trim();

    match input {
        "w" => Some(Action::Up),
        "s" => Some(Action::Down),
        "a" => Some(Action::Left),
        "d" => Some(Action::Right),
        "r" => Some(Action::Reset),
        "q" => Some(Action::Exit),
        _ => None,
    }
}

//-----------------------------------------------------
// Step Function
//-----------------------------------------------------

fn step(grid: &mut grid::Grid, agent_pos: Position, action: Action) -> StepResult {

    // Figure out the new row/col based on `action`
    let new_pos = match action{
        Action::Up    => Position { row: if agent_pos.row > 0 { agent_pos.row - 1 } else { agent_pos.row }, col: agent_pos.col },
        Action::Down  => Position { row: if agent_pos.row < grid.height - 1 { agent_pos.row + 1 } else { agent_pos.row }, col: agent_pos.col },
        Action::Left  => Position { row: agent_pos.row, col: if agent_pos.col > 0 { agent_pos.col - 1 } else { agent_pos.col } },
        Action::Right => Position { row: agent_pos.row, col: if agent_pos.col < grid.width - 1 { agent_pos.col + 1 } else { agent_pos.col } },
        Action::Reset => agent_pos,
        Action::Exit  => agent_pos,
    };

    // Calculate reward based on the new position
    let reward = if grid.data[new_pos.row][new_pos.col] == grid::Cell::Goal { 1.0 } else { 0.0 };
    let done = grid.data[new_pos.row][new_pos.col] == grid::Cell::Goal;

    // Update the grid: set old position back to Cell::Empty, new position to Cell::Agent
    grid.data[agent_pos.row][agent_pos.col] = grid::Cell::Empty;
    grid.data[new_pos.row][new_pos.col] = grid::Cell::Agent;

    // Return the new position, reward, and done status
    StepResult {
        position: new_pos,
        reward: reward, 
        done: done, 
    }
    
}

//-----------------------------------------------------
// Main Function
//-----------------------------------------------------

fn main() {

    let agent_start = Position { row: 0, col: 0 };
    let goal_start: Position = Position { row: GRID_SIZE - 1, col: GRID_SIZE - 1 };
    let mut grid: grid::Grid = grid::create_grid(GRID_SIZE, GRID_SIZE, (agent_start.row, agent_start.col), (goal_start.row, goal_start.col));
    let mut agent_pos = agent_start;

    // Reset grid with agent and goal positions
    grid::reset_grid(&mut grid);
    grid::print_grid(&grid);

    let mut tick: u32 = 0;

    loop {

        // Print the current tick
        println!("--- Tick {} ---", tick);

        // Read user input for action
        let action = match read_input() {
            Some(a) => a,
            None => {
                println!("Invalid input. Please enter w/a/s/d/r/q.");
                continue;
            }
        };

        // Handle the Reset action
        if action == Action::Reset {
            grid::reset_grid(&mut grid);
            agent_pos = Position { row: grid.agent_start.0, col: grid.agent_start.1 };
            println!("Grid has been reset.");
            grid::print_grid(&grid);
            tick += 1;
            continue;
        }

        // Handle the Exit action
        if action == Action::Exit {
            println!("Exiting the program.");
            break;
        }

        let step_result = step(&mut grid, agent_pos, action);
        agent_pos = step_result.position;

        grid::print_grid(&grid);
        println!("Reward: {}, Done: {}", step_result.reward, step_result.done);

        if step_result.done {
            println!("Goal reached at tick {}!", tick);
            break;
        }

        tick += 1;

    }

    println!("Total ticks passed: {}", tick);

}

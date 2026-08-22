//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

mod grid;
//-----------------------------------------------------
// Global Constants
//-----------------------------------------------------

const GRID_SIZE: usize = 10;

#[derive(Debug, Clone, Copy)]
enum Action {
    Up,
    Down,
    Left,
    Right,
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

    print!("Enter action (w/a/s/d): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let input = input.trim();

    match input {
        "w" => Some(Action::Up),
        "s" => Some(Action::Down),
        "a" => Some(Action::Left),
        "d" => Some(Action::Right),
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

    
    let mut loop_count = 0;
    let mut agent_pos = Position { row: 0, col: 0 };
    let mut grid: grid::Grid = grid::create_grid(GRID_SIZE, GRID_SIZE);
    grid.data[agent_pos.row][agent_pos.col] = grid::Cell::Agent;
    grid.data[GRID_SIZE - 1][GRID_SIZE - 1] = grid::Cell::Goal;
    
    grid::print_grid(&grid);

    while loop_count <= 100 {
        if let Some(action) = read_input() {
            let step_result = step(&mut grid, agent_pos, action);
            agent_pos = step_result.position; // Update agent position for the next step

            grid::print_grid(&grid);
            println!("New Agent Position: {:?}", step_result.position);
            println!("Reward: {}", step_result.reward);
            println!("Done: {}", step_result.done);
            
            if step_result.done {
                println!("Goal reached!");
                break;
            }
            loop_count += 1;
        } else {
            println!("Invalid input. Please enter w/a/s/d.");
        }
        
    }
    
    println!("Total steps taken: {}", loop_count);

}

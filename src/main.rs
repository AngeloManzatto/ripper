//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

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
    position: (
        usize, usize
    ),
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

fn step(grid: &mut grid::Grid, entity_id: u32, action: Action) -> StepResult {

    // Find the entity's CURRENT position by id.
    // Hint: grid.entities.iter().find(|e| e.id == entity_id) — same pattern as print_grid,
    // but this time you also need its .pos, and you'll need to handle the "not found" case
    // (for now, .expect("entity not found") is fine — a real recovery strategy can come later)
    let entity = grid.entities.iter().find(|e| e.id == entity_id).expect("entity not found");
    let current_pos: (usize, usize) = entity.pos;

    // Figure out the new row/col based on `action` — same logic as before,
    // but working with (usize, usize) tuples instead of Position now
    let new_pos: (usize, usize) = match action {
        Action::Up    => (if current_pos.0 > 0 {current_pos.0 - 1 } else {current_pos.0}, current_pos.1),
        Action::Down  => (if current_pos.0 < grid.height - 1 {current_pos.0 + 1 } else {current_pos.0}, current_pos.1),
        Action::Left  => (current_pos.0, if current_pos.1 > 0 {current_pos.1 - 1} else {current_pos.1}),
        Action::Right => (current_pos.0, if current_pos.1 < grid.width - 1 {current_pos.1 + 1} else {current_pos.1}),
        Action::Reset => current_pos,
        Action::Exit  => current_pos,
    };

    // Check if new_pos lands on any Goal-kind entity — search grid.entities again,
    // this time for kind == EntityKind::Goal && pos == new_pos
    let done: bool = grid.entities.iter().any(|e| e.kind == grid::EntityKind::Goal && e.pos == new_pos);
    let reward: f64 = if done { 1.0 } else { 0.0 };

    // Move the entity
    if let Some(moving_entity) = grid.entities.iter_mut().find(|e| e.id == entity_id) {
        moving_entity.pos = new_pos;
    }

    StepResult {
        position: new_pos,
        reward,
        done,
    }
}

//-----------------------------------------------------
// Main Function
//-----------------------------------------------------

fn main() {

    let agent_start =  (0, 0);
    let goal_pos = (GRID_SIZE - 1, GRID_SIZE - 1);
    let enemy_start = (5, 5);
    let config = grid::GridConfig{agent_start, goal_pos, enemy_start };
    let mut grid: grid::Grid = grid::create_grid(GRID_SIZE, GRID_SIZE, config);

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

        let entity_id = grid.player_id.expect("entity not found");
        let step_result = step(&mut grid, entity_id, action);

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

//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use ripper::grid;

//-----------------------------------------------------
// Global Constants
//-----------------------------------------------------

const GRID_SIZE: usize = 10;

//-----------------------------------------------------
// Read input
//-----------------------------------------------------
fn read_input() -> Option<grid::Action> {
    use std::io::{self, Write};

    print!("Enter action (w/a/s/d/r/q): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let input = input.trim();

    match input {
        "w" => Some(grid::Action::Up),
        "s" => Some(grid::Action::Down),
        "a" => Some(grid::Action::Left),
        "d" => Some(grid::Action::Right),
        "r" => Some(grid::Action::Reset),
        "q" => Some(grid::Action::Exit),
        _ => None,
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
        if action == grid::Action::Reset {
            grid::reset_grid(&mut grid);
            println!("Grid has been reset.");
            grid::print_grid(&grid);
            tick += 1;
            continue;
        }

        // Handle the Exit action
        if action == grid::Action::Exit {
            println!("Exiting the program.");
            break;
        }

        // Advance player and enemy
        let entity_id = grid.player_id.expect("entity not found");
        let step_result = grid::step(&mut grid, entity_id, action);
        grid::step_enemy(&mut grid);

        // Collision detection resolution
        if grid::check_collision(&grid) {
            grid::reset_grid(&mut grid);
            println!("GAME OVER!");
            grid::print_grid(&grid);
            tick += 1;
            continue;
        }

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

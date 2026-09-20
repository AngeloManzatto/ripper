//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use std::io::Write;

use crate::grid_world::world;
use crate::grid_world::vision;

//-----------------------------------------------------
// Trace Line 
//-----------------------------------------------------

pub fn get_cell_char_by_position(world: &world::World, pos: (usize, usize)) -> char {

    let found_id = world.positions.iter()
        .find(|(_, p)| **p == pos)
        .map(|(id, _)| *id);

    match found_id {
        Some(id) => {
            let kind = world.kinds.get(&id).expect("entity has no kind");
            kind.to_char()
        }
        None => '.',
    }
}

//-----------------------------------------------------
// Print world
//-----------------------------------------------------

pub fn render_world(world: &world::World) -> String {
    let mut output = String::new();

    for row in 0..world.height {
        for col in 0..world.width {
            let found_id = world.positions.iter()
                .find(|(_, pos)| **pos == (row, col))
                .map(|(id, _)| *id);

            let ch = match found_id {
                Some(id) => {
                    let kind = world.kinds.get(&id).expect("entity has no kind");
                    kind.to_char()
                }
                None => '.',
            };

            output.push(ch);
        }
        output.push('\n');
    }

    output
}

pub fn print_world(world: &world::World) {
    print!("{}", render_world(world));
    std::io::stdout().flush().unwrap();
}

//-----------------------------------------------------
// Print visible cells
//-----------------------------------------------------

pub fn print_visible_cells(world: &world::World, from: (usize, usize), perception_range: usize) {
    let visible_cells = vision::get_visible_cells(world, from, perception_range);
    for row in 0..world.height {
        let mut line = String::new();
        for col in 0..world.width {
            if visible_cells.contains(&(row, col)) {
                let ch = get_cell_char_by_position(world, (row, col));
                line.push(ch); // placeholder — real character lookup goes here
            } else {
                line.push('X');
            }
        }
        println!("{}", line);
    }
}

//************************************************************************************
// Tests
//************************************************************************************

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_world::layout;
    use crate::grid_world::world;

    fn test_world() -> world::World {
        let layout_str =
        ".......
        .......
        ...#...
        ..#....
        .......
        .......
        .......";

        let config = world::WorldConfig::default();
        layout::parse_layout(layout_str, &config)
    }

    #[test]
    fn wall_blocks_direct_line() {
        let world = test_world();
        print_world(&world);
        println!();
        print_visible_cells(&world, (3,3), 10);
    }
}
//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use std::io::Write;
use std::collections::HashSet;

use crate::grid_world::world;
use crate::grid_world::vision;
use crate::grid_world::entity::EntityKind;

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

//-----------------------------------------------------
// Find occupant kind (terrain-only lookup, no char conversion)
//-----------------------------------------------------

fn find_occupant_kind(world: &world::World, pos: (usize, usize)) -> Option<EntityKind> {
    world.positions.iter()
        .find(|(_, p)| **p == pos)
        .and_then(|(id, _)| world.kinds.get(id))
        .copied()
}

//-----------------------------------------------------
// Render entity view (FOW-aware, single entity's perspective)
//-----------------------------------------------------

pub fn render_entity_view(world: &world::World, entity_id: u32) -> String {
    let pos = world.positions[&entity_id];
    let perception_range = world.perception_ranges[&entity_id];
    let discovered = &world.discovered[&entity_id];

    let visible_cells: HashSet<(usize, usize)> =
        vision::get_visible_cells(world, pos, perception_range)
            .into_iter()
            .collect();

    let mut output = String::new();

    for row in 0..world.height {
        for col in 0..world.width {
            let cell = (row, col);

            let ch = if visible_cells.contains(&cell) {
                // Currently visible: show exactly what's there right now,
                // mover or terrain or floor.
                get_cell_char_by_position(world, cell)
            } else if discovered.contains(&cell) {
                // Discovered but not currently visible: terrain memory only,
                // movers never render here (matches observation()'s Pass 1).
                match find_occupant_kind(world, cell) {
                    Some(kind) if matches!(kind, EntityKind::Wall | EntityKind::Trap | EntityKind::Goal) => {
                        kind.to_char()
                    }
                    _ => '.',
                }
            } else {
                'X' // never discovered
            };

            output.push(ch);
        }
        output.push('\n');
    }

    output
}

pub fn print_entity_view(world: &world::World, entity_id: u32) {
    print!("{}", render_entity_view(world, entity_id));
    std::io::stdout().flush().unwrap();
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
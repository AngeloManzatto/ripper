//-----------------------------------------------------
// Imports
//-----------------------------------------------------

//-----------------------------------------------------
// Parse Layout
//-----------------------------------------------------

use crate::grid_world::entity::EntityKind;
use crate::grid_world::world::{World, spawn_entity};

pub fn parse_layout(layout: &str, max_tick:u32) -> World {

    // Initialize string of lines iterator
    let lines: Vec<&str> = layout.lines().collect();

    // Get grid size
    let height = lines.len();
    let width = lines[0].trim().len();

    // Initalize world
    let mut world = World {
        width,
        height,
        next_id: 0,
        positions: std::collections::HashMap::new(),
        kinds: std::collections::HashMap::new(),
        layout: layout.to_string(),
        done: false,
        end_reason: None,
        tick:0,
        max_tick: max_tick
    };

    for (row, line) in lines.iter().enumerate() {

        let trimmed_line = &line.trim();

        if trimmed_line.len() != width {
            panic!("Ragged layout: row {} has length {}, expected {}", row, line.len(), width);
        }

        for (col, ch) in trimmed_line.chars().enumerate() {
            let kind = match ch {
                '#' => Some(EntityKind::Wall),
                'P' => Some(EntityKind::Player),
                'E' => Some(EntityKind::Enemy),
                'G' => Some(EntityKind::Goal),
                'T' => Some(EntityKind::Trap),
                '.' => None,
                _ => panic!("Unknown layout character: '{}'", ch),
            };

            if let Some(k) = kind {
                spawn_entity(&mut world, k, (row, col));
            }
        }
    }

    world
}
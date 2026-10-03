//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use crate::grid_world::ecs::Ecs;
use crate::grid_world::entity::{EntityKind, TileKind};
use crate::grid_world::layout::parsers;
use crate::grid_world::world::state::{World, WorldConfig};

//-----------------------------------------------------
// World from layout
//-----------------------------------------------------

impl World {
    pub fn from_layout(layout: &str, config: &WorldConfig) -> World {
        // Initialize id counter
        let mut next_id: u32 = 0;

        // Initialize string of lines iterator
        let layout_grid = parsers::layout_to_grid(layout);

        // Get grid size
        let height = layout_grid.len();
        let width = layout_grid[0].len();

        // Initialize Entity Component System
        let mut ecs = Ecs::default();

        // Initialize Grid System
        let mut tile_grid = vec![vec![TileKind::Free; width]; height];

        for row in 0..height {
            for col in 0..width {
                // Get current chart
                let ch = layout_grid[row][col];

                // Check if the char is a tile type
                if let Some(tile_kind) = TileKind::from_char(ch) {
                    tile_grid[row][col] = tile_kind;
                } else {
                    // Check if char is an entity type
                    let Some(entity_kind) = EntityKind::from_char(ch) else {
                        panic!("Unknown layout character: '{}' at ({}, {})", ch, row, col);
                    };

                    let perception_range = match entity_kind {
                        EntityKind::Player => config.player_perception_range,
                        EntityKind::Enemy => config.enemy_perception_range,
                    };

                    ecs.spawn(next_id, entity_kind, (row, col), perception_range);
                    next_id += 1;

                    tile_grid[row][col] = TileKind::Free;
                }
            }
        }

        // Initalize world
        World {
            width,
            height,
            next_id,
            ecs: ecs,
            grid: tile_grid,
            layout: layout.to_string(),
            done: false,
            tick: 0,
            config: config.clone(),
        }
    }
}

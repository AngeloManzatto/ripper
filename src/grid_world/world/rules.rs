//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use crate::grid_world::entity::{EntityStatus, TileKind};
use crate::grid_world::world::state::World;

//-----------------------------------------------------
// Acessors
//-----------------------------------------------------

impl World {
    pub fn find_tiles_position_by_kind(&self, target: TileKind) -> Vec<(usize, usize)> {
        let mut positions = Vec::new();

        for row in 0..self.height {
            for col in 0..self.width {
                let tile_kind = self.grid[row][col];

                if tile_kind == target {
                    positions.push((row, col));
                }
            }
        }

        positions
    }
}

//-----------------------------------------------------
// Is walkable
//-----------------------------------------------------

impl World {
    pub fn is_walkable(&self, pos: (usize, usize)) -> bool {
        // Check if position is not a wall or something that prevents walking into it
        // Panics if `pos` is out of bounds.
        self.grid[pos.0][pos.1] != TileKind::Wall
    }
}

//-----------------------------------------------------
// Is free
//-----------------------------------------------------

impl World {
    pub fn is_free(&self, pos: (usize, usize)) -> bool {
        // Check if position is empty
        self.grid[pos.0][pos.1] == TileKind::Free
    }
}

//-----------------------------------------------------
// Is goal
//-----------------------------------------------------

impl World {
    pub fn is_goal(&self, pos: (usize, usize)) -> bool {
        // Check if position is goal
        self.grid[pos.0][pos.1] == TileKind::Goal
    }
}

//-----------------------------------------------------
// Clamp move
//-----------------------------------------------------

impl World {
    pub fn clamp_move(&self, pos: (usize, usize), delta: (i32, i32)) -> (usize, usize) {
        // Avoid letting a moving entity going out of grid bounds
        let row = (pos.0 as i32 + delta.0).clamp(0, self.height as i32 - 1);
        let col = (pos.1 as i32 + delta.1).clamp(0, self.width as i32 - 1);
        (row as usize, col as usize)
    }
}

//-----------------------------------------------------
// Check goal
//-----------------------------------------------------

impl World {
    pub fn check_goal(&self, player_id: u32) -> bool {
        // Get player position
        let player_pos = self.ecs.position_of(player_id);

        // Check if player position is the same as goal position
        self.grid[player_pos.0][player_pos.1] == TileKind::Goal
    }
}

//-----------------------------------------------------
// Check trap
//-----------------------------------------------------

impl World {
    pub fn check_trap_collision(&self, entity_id: u32) -> bool {
        // Get entity position
        let entity_pos = self.ecs.position_of(entity_id);

        // Check if entity position is the same as trap position
        self.grid[entity_pos.0][entity_pos.1] == TileKind::Trap
    }
}

//-----------------------------------------------------
// Check enemy collision
//-----------------------------------------------------

impl World {
    pub fn check_enemy_collision(&self, player_id: u32) -> bool {
        // Get player position
        let player_pos = self.ecs.position_of(player_id);

        let enemy_ids = self.ecs.enemy_ids();

        for enemy_id in enemy_ids {

            if self.ecs.status_of(enemy_id) != EntityStatus::Alive {
                continue;
            }

            if player_pos == self.ecs.position_of(enemy_id) {
                return true;
            }
        }

        false
    }
}

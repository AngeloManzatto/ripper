//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use rand::Rng;
use rand::seq::SliceRandom;

use crate::grid_world::entity::TileKind;
use crate::grid_world::layout::parsers;
use crate::grid_world::pathfinding::{bfs, geometry};
use crate::grid_world::world::state::World;

//-----------------------------------------------------
// Walkable floor cells
//-----------------------------------------------------

impl World {
    fn find_free_cells(&self) -> Vec<(usize, usize)> {
        let mut cells = Vec::new();
        for row in 0..self.height {
            for col in 0..self.width {
                if self.is_free((row, col)) {
                    cells.push((row, col));
                }
            }
        }
        cells
    }
}

//-----------------------------------------------------
// Reposition entities
//-----------------------------------------------------

impl World {
    pub fn reposition_entities(&mut self, rng: &mut impl Rng) -> Option<()> {
        const MAX_ATTEMPTS: usize = 20; // tune later

        let grid = parsers::layout_to_grid(&self.layout);

        // ECS positions
        let player_id = self.ecs.player_id();
        let enemy_ids = self.ecs.enemy_ids();

        // Get candidate cells for mutation
        let mut candidate_cells = self.find_free_cells();
        let goal_cell = self.find_tiles_position_by_kind(TileKind::Goal)[0];

        // Goal cell itself can be a possibility here
        candidate_cells.push(goal_cell);

        // Total entities that can be repositioned
        let n_needed = enemy_ids.len() + 2; // Player, Goal, (2) + Enemies

        for _attempt in 0..MAX_ATTEMPTS {
            let mut chosen: Vec<_> = candidate_cells
                .choose_multiple(rng, n_needed)
                .copied()
                .collect();

            // not enough walkable cells at all, retrying won't help
            if chosen.len() < n_needed {
                return None;
            };

            let candidate_player_cell = chosen.pop()?;
            let candidate_goal_cell = chosen.pop()?;

            // Check valid path  between player and goal
            let Some(_valid_path) = bfs::bfs_search(
                grid.clone(),
                candidate_player_cell,
                candidate_goal_cell,
                vec!['#', 'T'],
            ) else {
                continue;
            };

            // Check minimum distance between player and goal
            if geometry::manhattan_distance(candidate_player_cell, candidate_goal_cell)
                >= self.config.min_player_goal_distance as i32
            {
                // Update player position on entity system
                self.ecs.set_position(player_id, candidate_player_cell);

                // Update goal on tile grid
                self.grid[candidate_goal_cell.0][candidate_goal_cell.1] = TileKind::Goal;
                self.grid[goal_cell.0][goal_cell.1] = TileKind::Free;

                for enemy_id in enemy_ids {
                    let enemy_pos = chosen
                        .pop()
                        .expect("chosen should have exactly enemy_ids.len() cells left");
                    self.ecs.set_position(enemy_id, enemy_pos);
                }

                return Some(());
            }
        }

        None // exhausted retries
    }
}

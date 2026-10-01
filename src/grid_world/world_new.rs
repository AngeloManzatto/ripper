//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use crate::grid_world::ecs::Ecs;
use crate::grid_world::entity::{TileKind};

//-----------------------------------------------------
// World Configuration
//-----------------------------------------------------

#[derive(Clone, Debug)]
pub struct WorldConfig {
    pub max_tick: u32,
    pub player_perception_range: usize,
    pub enemy_perception_range: usize,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
            max_tick: 100,
            player_perception_range: 5,
            enemy_perception_range: 5,
        }
    }
}

//-----------------------------------------------------
// World
//-----------------------------------------------------

pub struct World {
    pub width: usize,
    pub height: usize,
    pub next_id: u32,
    pub ecs: Ecs,
    pub layout: String,
    pub tick: u32,
    pub config: WorldConfig,
    pub done: bool,
    pub grid: Vec<Vec<TileKind>>,
}

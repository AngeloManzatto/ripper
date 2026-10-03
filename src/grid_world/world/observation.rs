//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use crate::grid_world::entity::EntityStatus;
use crate::grid_world::entity::{EntityKind, TileKind};
use crate::grid_world::world::state::World;
use crate::grid_world::world::perception::PerceivedCell;

//-----------------------------------------------------
// Cell encoding
//-----------------------------------------------------

fn entity_kind_to_cell(kind: EntityKind) -> i32 {
    match kind {
        EntityKind::Player => 2,
        EntityKind::Enemy => 3,
    }
}

fn tile_kind_to_cell(kind: TileKind) -> i32 {
    match kind {
        TileKind::Free => 0,
        TileKind::Wall => 1,
        TileKind::Trap => 4,
        TileKind::Goal => 5,
    }
}

//-----------------------------------------------------
// Entity Observation
//-----------------------------------------------------

pub struct EntityObservation {
    pub id: u32,
    pub position: (usize, usize),
    pub grid: Vec<Vec<i32>>,
    pub status: EntityStatus,
}

//-----------------------------------------------------
// Observation
//-----------------------------------------------------

pub struct Observation {
    pub entities: Vec<EntityObservation>,
    pub player_id: u32,
    pub enemy_ids: Vec<u32>,
}

//-----------------------------------------------------
// Observation
//-----------------------------------------------------

pub struct TimeStep {
    pub observation: Observation,
    pub terminated: bool,
    pub truncated: bool,
}

impl TimeStep {
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

//-----------------------------------------------------
// Observation
//-----------------------------------------------------

impl World {
    pub fn observation(&self) -> Observation {

        let entities: Vec<EntityObservation> = self.ecs.ids().map(|id| {

            // Get entity perceived grid
            let perceived = self.perceived_grid(id);

            // 
            let grid: Vec<Vec<i32>> = perceived.iter().map(|row| {
                row.iter().map(|cell| match cell {
                    PerceivedCell::Visible { occupant: Some(kind), .. } => entity_kind_to_cell(*kind),
                    PerceivedCell::Visible { terrain, occupant: None } => tile_kind_to_cell(*terrain),
                    PerceivedCell::Remembered { terrain } => tile_kind_to_cell(*terrain),
                    PerceivedCell::Unknown => -1,
                }).collect()
            }).collect();

            EntityObservation {
                id,
                position: self.ecs.position_of(id),
                grid,
                status: self.ecs.status_of(id),
            }

        }).collect();

        Observation {
            entities,
            player_id: self.ecs.player_id(),
            enemy_ids: self.ecs.enemy_ids(),
        }
    }
}

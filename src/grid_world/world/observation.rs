//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use std::collections::{HashMap, HashSet};

use crate::grid_world::entity::{EntityKind, TileKind};
use crate::grid_world::world::state::World;
use crate::grid_world::entity::{EntityStatus};

//-----------------------------------------------------
// Cell encoding
//-----------------------------------------------------

fn entity_kind_to_cell(kind: EntityKind) -> i32 {
    match kind {
        EntityKind::Player => 2,
        EntityKind::Enemy  => 3,
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
    pub status: EntityStatus
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
    pub observation : Observation,
    pub terminated : bool,
    pub truncated : bool,
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
    pub fn observation(&self) -> Observation 
    {

        // Build once: position -> id, for every live entity
        let occupant_at: HashMap<(usize, usize), u32> = self.ecs.ids()
            .map(|id| (self.ecs.position_of(id), id))
            .collect();

        let entities: Vec<EntityObservation> = self.ecs.ids().map(|id| {
            let pos = self.ecs.position_of(id);
            let perception_range = self.ecs.perception_of(id);
            let discovered = self.ecs.discovered_of(id);
            let visible_cells: HashSet<(usize, usize)> =
                self.get_visible_cells(pos, perception_range).into_iter().collect();

            let mut grid = vec![vec![-1i32; self.width]; self.height];

            // Pass 1: terrain — every discovered cell, straight from self.grid now
            for &(r, c) in discovered {
                grid[r][c] = tile_kind_to_cell(self.grid[r][c]);
            }

            // Pass 2: movers — only currently-visible cells, using the reverse lookup
            for &(r, c) in &visible_cells {
                if let Some(&occ_id) = occupant_at.get(&(r, c)) {
                    grid[r][c] = entity_kind_to_cell(self.ecs.kind_of(occ_id));
                }
            }

            EntityObservation { id, position: pos, grid, status: self.ecs.status_of(id) }
        }).collect();

        Observation {
            entities,
            player_id: self.ecs.player_id(),
            enemy_ids: self.ecs.enemy_ids(),
        }
    }
}
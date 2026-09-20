//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use rand;
use std::collections::HashMap;
use std::usize;

use crate::grid_world::entity::EntityKind;
use crate::grid_world::action::Action;
use crate::grid_world::layout::{parse_layout};
use crate::grid_world::environment::Environment;
use crate::grid_world::vision;

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
// Observation
//-----------------------------------------------------

pub struct EntityObservation {
    pub id: u32,
    pub position: (usize, usize),
    pub grid: Vec<Vec<i32>>,
}

pub struct Observation {
    pub entities: Vec<EntityObservation>,
    pub player_id: u32,
    pub enemy_ids: Vec<u32>,
}

//-----------------------------------------------------
// Cell encoding
//-----------------------------------------------------

fn kind_to_cell(kind: EntityKind) -> i32 {
    match kind {
        EntityKind::Wall   => 1,
        EntityKind::Player => 2,
        EntityKind::Enemy  => 3,
        EntityKind::Trap   => 4,
        EntityKind::Goal   => 5,
    }
}

//-----------------------------------------------------
// Step Result 
//-----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    GoalReached,
    Caught,
    Trapped,
    Timeout,
}

pub struct StepResult {
    pub done: bool,
    pub reason: Option<EndReason>,
}

//-----------------------------------------------------
// World
//-----------------------------------------------------

pub struct World {
    pub width: usize,
    pub height: usize,
    pub next_id: u32,
    pub positions  : HashMap<u32, (usize, usize)>,
    pub kinds      : HashMap<u32, EntityKind>,
    pub discovered : HashMap<u32, std::collections::HashSet<(usize, usize)>>,
    pub perception_ranges: HashMap<u32, usize>,
    pub layout: String,
    pub done: bool,
    pub end_reason: Option<EndReason>,
    pub tick: u32,
    pub config:WorldConfig
}

//-----------------------------------------------------
// Environment
//-----------------------------------------------------

impl Environment for World {
    
    type Action = Action;
    type Observation = Observation;

    // Reset world
    fn reset(&mut self) {
        let new_world = parse_layout(&self.layout, &self.config);
        *self = new_world;
    }

    fn step(&mut self, actions: HashMap<u32, Action>) -> StepResult {

        //--------------------------------------------------
        // Done resolution
        //--------------------------------------------------

        if self.done {
            return StepResult { done: self.done, reason: self.end_reason };
        }

        //--------------------------------------------------
        // Time out resolution
        //--------------------------------------------------

        // Increment the tick counter; check if tick >= max_tick → timeout.
        self.tick += 1;

        // Check if episode is finished by time out
        if self.tick >= self.config.max_tick {
            self.done = true;
            self.end_reason = Some(EndReason::Timeout);
            return StepResult { done: self.done, reason: self.end_reason };
        }

        //--------------------------------------------------
        // Movement resolution
        //--------------------------------------------------

        // Resolve the final action for every moving entity (player + all enemies)
        let mut resolved_actions: HashMap<u32, Action> = HashMap::new();

        let player_id = get_entity_ids_by_kind(self, EntityKind::Player)[0];
        resolved_actions.insert(player_id, *actions.get(&player_id).expect("player action required"));
        
        for enemy_id in get_entity_ids_by_kind(self, EntityKind::Enemy) {
            let action = actions.get(&enemy_id).copied()
                .unwrap_or_else(|| rand::random()); // internal fallback
            resolved_actions.insert(enemy_id, action);
        }

        // Apply movement uniformly, no player/enemy distinction needed
        for (&id, &action) in resolved_actions.iter() {
            let current_pos = self.positions[&id];
            let delta = action_to_delta(action);
            let new_pos = clamp_move(self, current_pos, delta);
            let final_pos = if is_walkable(self, new_pos) { new_pos } else { current_pos };
            self.positions.insert(id, final_pos);
        }

        //--------------------------------------------------
        // Collision resolution
        //--------------------------------------------------

        // Check if any entity steppen inside a trap
        for (&id, _) in resolved_actions.iter() {
            if check_trap_collision(self, id) {
                if id == player_id {
                    self.done = true;
                    self.end_reason = Some(EndReason::Trapped);
                } else {
                    // Eliminate enemy from the map.
                    despawn_entity(self, id);
                }
            }
        }

        // Check enemy collision (only matters if player is still alive)
        if check_enemy_collision(self, player_id) {
            self.done = true;
            self.end_reason = Some(EndReason::Caught);
        }

        // Check goal (only matters if player is still alive)
        if check_goal(self, player_id) {
            self.done = true;
            self.end_reason = Some(EndReason::GoalReached);
        }

        //--------------------------------------------------
        // Update discovered cells (FOW)
        //--------------------------------------------------

        for (&id, _) in resolved_actions.iter() {
            if let Some(&perception_range) = self.perception_ranges.get(&id) {
                let current_pos = self.positions[&id];
                update_discovered(self, id, current_pos, perception_range);
            }
        }

        StepResult { done: self.done, reason: self.end_reason }
    }

   fn observation(&self) -> Observation {
    
        let entities: Vec<EntityObservation> = self.positions.iter()
            .filter_map(|(&id, &pos)| {
                let discovered = self.discovered.get(&id)?;
                let perception_range = *self.perception_ranges.get(&id)?;

                // This tick's actual FOV — recomputed fresh, not the
                // accumulated `discovered` set. Mirrors draw_world's
                // `visible_cells` argument.
                let visible_cells: std::collections::HashSet<(usize, usize)> =
                    vision::get_visible_cells(self, pos, perception_range)
                        .into_iter()
                        .collect();

                let mut grid = vec![vec![-1i32; self.width]; self.height];

                // Pass 1: terrain — shown if ever discovered, regardless of
                // current visibility. Movers occupying a merely-discovered
                // (not currently visible) cell are NOT drawn here.
                for &(r, c) in discovered {
                    grid[r][c] = self.positions.iter()
                        .find_map(|(k, &p)| if p == (r, c) { Some(k) } else { None })
                        .map(|occ_id| self.kinds[occ_id])
                        .filter(|kind| matches!(kind, EntityKind::Wall | EntityKind::Trap | EntityKind::Goal))
                        .map(kind_to_cell)
                        .unwrap_or(0); // empty floor
                }

                // Pass 2: movers — only shown if currently visible this tick,
                // overwriting whatever pass 1 wrote (floor or terrain-under-them).
                for &(r, c) in &visible_cells {
                    if let Some(occ_id) = self.positions.iter()
                        .find_map(|(k, &p)| if p == (r, c) { Some(k) } else { None })
                    {
                        if matches!(self.kinds[occ_id], EntityKind::Player | EntityKind::Enemy) {
                            grid[r][c] = kind_to_cell(self.kinds[occ_id]);
                        }
                    }
                }

                Some(EntityObservation { id, position: pos, grid })
            })
            .collect();

        let player_id = get_entity_ids_by_kind(self, EntityKind::Player)[0];
        let enemy_ids = get_entity_ids_by_kind(self, EntityKind::Enemy);

        Observation { entities, player_id, enemy_ids }
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

//-----------------------------------------------------
// Spawn Entity
//-----------------------------------------------------

pub fn spawn_entity(world: &mut World, kind: EntityKind, pos: (usize, usize)) -> u32 {
    let id = world.next_id;
    world.next_id += 1;

    world.positions.insert(id, pos);
    world.kinds.insert(id, kind);

    id
}

//-----------------------------------------------------
// Depawn Entity
//-----------------------------------------------------
pub fn despawn_entity(world: &mut World, id: u32) {
    // Remove all ids from ECS related to this entity

    world.kinds.remove(&id).unwrap_or_else(|| panic!("ID '{}' not found in kinds", id));
    world.positions.remove(&id).unwrap_or_else(|| panic!("ID '{}' not found in positions", id));
    world.discovered.remove(&id).unwrap_or_else(|| panic!("ID '{}' not found in discovered", id));
    world.perception_ranges.remove(&id).unwrap_or_else(|| panic!("ID '{}' not found in perception_range", id));
}

//-----------------------------------------------------
// Getters 
//-----------------------------------------------------

pub fn get_entity_ids_by_kind(world: &World, kind: EntityKind) -> Vec<u32>
{

    world.kinds.iter()
               .filter(|&(_, _kind)| *_kind == kind)
               .map(|(&id, _)| id)
               .collect()

}

//-----------------------------------------------------
// Action to delta 
//-----------------------------------------------------
fn action_to_delta(action: Action) -> (i32, i32)
{
    match action {
        Action::Up    => (-1, 0),
        Action::Down  => (1, 0),
        Action::Left  => (0, -1),
        Action::Right => (0, 1),
    }
}

//-----------------------------------------------------
// Clamp move
//-----------------------------------------------------
pub fn clamp_move(world: &World, pos: (usize, usize), delta: (i32, i32)) -> (usize, usize) {

    // Clamp delta inside the grid boundaries
    let row = (pos.0 as i32 + delta.0).clamp(0, world.height as i32 - 1);
    let col = (pos.1 as i32 + delta.1).clamp(0, world.width as i32 - 1);
    (row as usize, col as usize)
}

//-----------------------------------------------------
// Is Walkable
//-----------------------------------------------------
pub fn is_walkable(world: &World, pos: (usize, usize)) -> bool {

    // Check if that position is already occupied by a wall
    !world.positions.iter().any(|(&id, &p)| {
        p == pos && world.kinds.get(&id) == Some(&EntityKind::Wall)
    })

}

//-----------------------------------------------------
// Check goal
//-----------------------------------------------------

pub fn check_goal(world: &World, player_id: u32) -> bool {

    // Get player position
    let player_pos = world.positions[&player_id];

    // Check if player position is the same as goal position
    world.positions.iter().any(|(&id, &pos)| {
        pos == player_pos && world.kinds.get(&id) == Some(&EntityKind::Goal)
    })
}

//-----------------------------------------------------
// Check enemy collision
//-----------------------------------------------------
pub fn check_enemy_collision(world: &World, player_id: u32) -> bool {

    // Get player position
    let player_pos = world.positions[&player_id];

    // Check if player
    world.positions.iter().any(|(&id, &pos)| {
        pos == player_pos && world.kinds.get(&id) == Some(&EntityKind::Enemy)
    })
}

//-----------------------------------------------------
// Check trap collision
//-----------------------------------------------------

pub fn check_trap_collision(world: &World, entity_id: u32) -> bool {

    // Get entity position
    let entity_pos = world.positions[&entity_id];

    // Check if player
    world.positions.iter().any(|(&id, &pos)| {
        pos == entity_pos && world.kinds.get(&id) == Some(&EntityKind::Trap)
    })
}

//-----------------------------------------------------
// Update discovered (FOW)
//-----------------------------------------------------

pub fn update_discovered(world: &mut World, id: u32, pos: (usize, usize), perception_range: usize)
{

    let visible = vision::get_visible_cells(world, pos, perception_range);

    world.discovered.entry(id)
                    .or_insert_with(std::collections::HashSet::new)
                    .extend(visible);
}
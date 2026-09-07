//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use rand;
use std::io::Write;
use std::collections::HashMap;
use crate::grid_world::entity::EntityKind;
use crate::grid_world::action::Action;
use crate::grid_world::layout::parse_layout;
use crate::grid_world::environment::Environment;

//-----------------------------------------------------
// Observation
//-----------------------------------------------------

pub struct Observation {
    pub grid: Vec<Vec<Vec<f64>>>,
    pub player_id: u32,
    pub enemy_ids: Vec<u32>,
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
    pub positions: HashMap<u32, (usize, usize)>,
    pub kinds: HashMap<u32, EntityKind>,
    pub layout: String,
    pub done: bool,
    pub end_reason: Option<EndReason>,
    pub tick: u32,
    pub max_tick:u32
}

//-----------------------------------------------------
// Environment
//-----------------------------------------------------

impl Environment for World {
    
    type Action = Action;
    type Observation = Observation;

    // Reset world
    fn reset(&mut self) {
        let new_world = parse_layout(&self.layout, self.max_tick);
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
        if self.tick >= self.max_tick {
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

        StepResult { done: self.done, reason: self.end_reason }
    }

   fn observation(&self) -> Observation {
        let mut grid = vec![vec![vec![0.0; self.width]; self.height]; 5];

        for (&id, &pos) in self.positions.iter() {
            let kind = self.kinds.get(&id).expect("entity has no kind");
            let channel = match kind {
                EntityKind::Player => 0,
                EntityKind::Enemy  => 1,
                EntityKind::Goal   => 2,
                EntityKind::Wall   => 3,
                EntityKind::Trap   => 4,
            };
            grid[channel][pos.0][pos.1] = 1.0;
        }

        let player_id = get_entity_ids_by_kind(self, EntityKind::Player)[0];
        let enemy_ids = get_entity_ids_by_kind(self, EntityKind::Enemy);

        Observation { grid, player_id, enemy_ids }
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

}

//-----------------------------------------------------
// Query entity ids by kind
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
// Print world
//-----------------------------------------------------

pub fn render_world(world: &World) -> String {
    let mut output = String::new();

    for row in 0..world.height {
        for col in 0..world.width {
            let found_id = world.positions.iter()
                .find(|(_, pos)| **pos == (row, col))
                .map(|(id, _)| *id);

            let ch = match found_id {
                Some(id) => {
                    let kind = world.kinds.get(&id).expect("entity has no kind");
                    match kind {
                        EntityKind::Player => 'P',
                        EntityKind::Enemy  => 'E',
                        EntityKind::Goal   => 'G',
                        EntityKind::Wall   => '#',
                        EntityKind::Trap   => 'T',
                    }
                }
                None => '.',
            };
            output.push(ch);
        }
        output.push('\n');
    }

    output
}

pub fn print_world(world: &World) {
    print!("{}", render_world(world));
    std::io::stdout().flush().unwrap();
}
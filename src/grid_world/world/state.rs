//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use rand::Rng;
use std::collections::HashMap;

use crate::grid_world::action::Action;
use crate::grid_world::ecs;
use crate::grid_world::entity::{EntityStatus, TileKind};
use crate::grid_world::world::observation::TimeStep;

//-----------------------------------------------------
// World Configuration
//-----------------------------------------------------

#[derive(Clone, Debug)]
pub struct WorldConfig {
    pub max_tick: u32,
    pub player_perception_range: usize,
    pub enemy_perception_range: usize,
    pub min_player_goal_distance: usize,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
            max_tick: 100,
            player_perception_range: 5,
            enemy_perception_range: 5,
            min_player_goal_distance: 0,
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
    pub ecs: ecs::Ecs,
    pub grid: Vec<Vec<TileKind>>,
    pub layout: String,
    pub tick: u32,
    pub config: WorldConfig,
    pub done: bool,
}

//-----------------------------------------------------
// Reset
//-----------------------------------------------------

impl World {
    pub fn reset(&mut self) -> TimeStep {

        // Rebuild world from layout
        *self = World::from_layout(&self.layout, &self.config);

        // Update entities views
        self.update_all_discovered();

        return TimeStep {
            observation: self.observation(),
            terminated: false,
            truncated: false,
        };
    }

    pub fn reset_with_reposition(&mut self, rng: &mut impl Rng) -> TimeStep 
    {
        // Reset world
        self.reset();

        // Reposition entities
        self.reposition_entities(rng);

        // Update entities views
        self.update_all_discovered();

        return TimeStep {
            observation: self.observation(),
            terminated: false,
            truncated: false,
        };
    }

    pub fn remove_dead(&mut self)
    {
        for enemy_id in self.ecs.enemy_ids(){

            if self.ecs.status_of(enemy_id) != EntityStatus::Alive {
                self.ecs.despawn(enemy_id);
            }
        }
    }

}

//-----------------------------------------------------
// Step
//-----------------------------------------------------

impl World {
    pub fn step(&mut self, actions: HashMap<u32, Action>, rng: &mut impl Rng) -> TimeStep {
        // Game Over
        if self.done {
            self.done = true;
            return TimeStep {
                observation: self.observation(),
                terminated: true,
                truncated: false,
            };
        }

        self.tick += 1;

        // Time over
        if self.tick >= self.config.max_tick {
            self.done = true;
            return TimeStep {
                observation: self.observation(),
                terminated: false,
                truncated: true,
            };
        }

        let player_id = self.ecs.player_id();
        let mut terminated = false;

        // 1. Resolve actions (player required, enemies fall back to a random action via `rng`)
        let mut resolved_actions: HashMap<u32, Action> = HashMap::new();
        resolved_actions.insert(
            player_id,
            *actions.get(&player_id).expect("player action required"),
        );

        for enemy_id in self.ecs.enemy_ids() {
            let action = actions
                .get(&enemy_id)
                .copied()
                .unwrap_or_else(|| rng.r#gen());
            resolved_actions.insert(enemy_id, action);
        }

        // 2. Apply movement for every resolved id (clamp_move + is_walkable check)
        for (&id, &action) in resolved_actions.iter() {
            let current_position = self.ecs.position_of(id);
            let delta = action.to_delta();
            let new_position = self.clamp_move(current_position, delta);

            let final_position = if id == player_id {
                if self.is_walkable(new_position) {
                    new_position
                } else {
                    current_position
                }
            } else {
                if self.is_walkable(new_position) && !self.is_goal(new_position) {
                    new_position
                } else {
                    current_position
                }
            };

            self.ecs.set_position(id, final_position);
        }

        // 3. Trap collision resolution
        for (&id, _) in resolved_actions.iter() {
            if self.check_trap_collision(id) {
                if id == player_id {
                    self.done = true;
                    terminated = true;
                }
                self.ecs.set_status(id, EntityStatus::Trapped);
            }
        }

        // 4. Goal / enemy-collision checks on the player
        if self.check_goal(player_id) {
            terminated = true;
            self.done = true;
            self.ecs.set_status(player_id, EntityStatus::GoalReached);
        }

        if self.check_enemy_collision(player_id) {
            terminated = true;
            self.done = true;
            self.ecs.set_status(player_id, EntityStatus::Caught);
        }

        // 5. FOW update (all entities still in the ECS, including ones that just died).
        for &id in resolved_actions.keys() {
            let pos = self.ecs.position_of(id);
            let perception_range = self.ecs.perception_of(id);
            self.update_discovered(id, pos, perception_range);
        }

        let observation = self.observation();

        // Remove dead entities
        self.remove_dead();

        TimeStep {
            observation: observation,
            terminated,
            truncated: false,
        }
    }
}

#[cfg(test)]
mod tests;
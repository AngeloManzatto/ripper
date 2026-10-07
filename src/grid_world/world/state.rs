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

        // 3. Trap collision: player -> self.done = true; enemy -> self.ecs.despawn(id)
        for (&id, _) in resolved_actions.iter() {
            if self.check_trap_collision(id) {
                if id == player_id {
                    self.done = true;
                    terminated = true;
                    self.ecs.set_status(id, EntityStatus::Trapped);
                } else {
                    // Eliminate enemy from the map.
                    self.ecs.despawn(id);
                }
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

        // 5. FOW update (only still-alive ids)
        for &id in resolved_actions.keys() {
            if self.ecs.ids().any(|alive_id| alive_id == id) {
                let pos = self.ecs.position_of(id);
                let perception_range = self.ecs.perception_of(id);
                self.update_discovered(id, pos, perception_range);
            }
        }

        TimeStep {
            observation: self.observation(),
            terminated,
            truncated: false,
        }
    }
}

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_world::entity::EntityStatus;
    use std::collections::HashMap;

    fn default_layout() -> &'static str {
        "#####\n\
         #P.T#\n\
         #...#\n\
         #..G#\n\
         #####"
    }

    fn layout_with_enemy() -> &'static str {
        "#####\n\
         #P.E#\n\
         #...#\n\
         #..G#\n\
         #####"
    }

    #[test]
    fn from_layout_builds_correct_dimensions_and_positions() {
        let world = World::from_layout(default_layout(), &WorldConfig::default());

        assert_eq!(world.width, 5);
        assert_eq!(world.height, 5);

        let player_id = world.ecs.player_id();
        assert_eq!(world.ecs.position_of(player_id), (1, 1));
        assert_eq!(
            world.find_tiles_position_by_kind(TileKind::Goal),
            vec![(3, 3)]
        );
        assert_eq!(
            world.find_tiles_position_by_kind(TileKind::Trap),
            vec![(1, 3)]
        );
    }

    #[test]
    fn step_moves_player_into_free_cell() {
        let mut world = World::from_layout(default_layout(), &WorldConfig::default());
        let player_id = world.ecs.player_id();
        let mut rng = rand::thread_rng();

        let mut actions = HashMap::new();
        actions.insert(player_id, Action::Right); // (1,1) -> (1,2), free

        let step = world.step(actions, &mut rng);

        assert_eq!(world.ecs.position_of(player_id), (1, 2));
        assert!(!step.terminated);
        assert!(!step.truncated);
        assert!(!world.done);
    }

    #[test]
    fn step_blocked_by_wall_leaves_player_in_place() {
        let mut world = World::from_layout(default_layout(), &WorldConfig::default());
        let player_id = world.ecs.player_id();
        let mut rng = rand::thread_rng();

        let mut actions = HashMap::new();
        actions.insert(player_id, Action::Left); // (1,1) -> (1,0), wall

        world.step(actions, &mut rng);

        assert_eq!(world.ecs.position_of(player_id), (1, 1));
    }

    #[test]
    fn step_player_reaches_goal_sets_done_and_terminated() {
        let mut world = World::from_layout(default_layout(), &WorldConfig::default());
        let player_id = world.ecs.player_id();
        let mut rng = rand::thread_rng();

        // Walk player from (1,1) to (3,3): down, down, right, right
        let path = [Action::Down, Action::Down, Action::Right, Action::Right];
        let mut step = None;
        for action in path {
            let mut actions = HashMap::new();
            actions.insert(player_id, action);
            step = Some(world.step(actions, &mut rng));
            if world.done {
                break;
            }
        }

        let step = step.expect("at least one step ran");
        assert!(world.done);
        assert!(step.terminated);
        assert!(!step.truncated);
        assert_eq!(world.ecs.status_of(player_id), EntityStatus::GoalReached);
    }

    #[test]
    fn step_player_hits_trap_sets_done_and_terminated() {
        let mut world = World::from_layout(default_layout(), &WorldConfig::default());
        let player_id = world.ecs.player_id();
        let mut rng = rand::thread_rng();

        // (1,1) -> (1,2) -> (1,3) is the trap
        for action in [Action::Right, Action::Right] {
            let mut actions = HashMap::new();
            actions.insert(player_id, action);
            world.step(actions, &mut rng);
        }

        assert!(world.done);
        assert_eq!(world.ecs.status_of(player_id), EntityStatus::Trapped);
    }

    #[test]
    fn step_timeout_sets_truncated() {
        let mut config = WorldConfig::default();
        config.max_tick = 1;
        let mut world = World::from_layout(default_layout(), &config);
        let player_id = world.ecs.player_id();
        let mut rng = rand::thread_rng();

        let mut actions = HashMap::new();
        actions.insert(player_id, Action::Right);

        let step = world.step(actions, &mut rng);

        assert!(world.done);
        assert!(step.truncated);
        assert!(!step.terminated);
    }

    #[test]
    fn step_enemy_collision_sets_done_and_caught() {
        let mut world = World::from_layout(layout_with_enemy(), &WorldConfig::default());
        let player_id = world.ecs.player_id();
        let enemy_id = world.ecs.enemy_ids()[0];
        let mut rng = rand::thread_rng();

        // Player (1,1) -> (1,2); Enemy (1,3) -> (1,2): collide
        let mut actions = HashMap::new();
        actions.insert(player_id, Action::Right);
        actions.insert(enemy_id, Action::Left);

        let step = world.step(actions, &mut rng);

        assert!(world.done);
        assert!(step.terminated);
        assert_eq!(world.ecs.status_of(player_id), EntityStatus::Caught);
    }

    #[test]
    fn reset_restores_original_spawn_positions() {
        let mut world = World::from_layout(default_layout(), &WorldConfig::default());
        let player_id = world.ecs.player_id();
        let mut rng = rand::thread_rng();

        let mut actions = HashMap::new();
        actions.insert(player_id, Action::Right);
        world.step(actions, &mut rng);
        assert_ne!(world.ecs.position_of(player_id), (1, 1)); // confirm it actually moved

        world.reset();

        let player_id_after = world.ecs.player_id();
        assert_eq!(world.ecs.position_of(player_id_after), (1, 1));
        assert!(!world.done);
        assert_eq!(world.tick, 0);
    }

    #[test]
    fn reposition_entities_returns_some_with_default_config() {
        let mut world = World::from_layout(default_layout(), &WorldConfig::default());
        let mut rng = rand::thread_rng();

        let result = world.reposition_entities(&mut rng);

        assert!(result.is_some());
    }

    #[test]
    fn reset_remembers_initial_view() {
        let mut world = World::from_layout(default_layout(), &WorldConfig::default());
        world.reset();
        let pid = world.ecs.player_id();
        let pos = world.ecs.position_of(pid);
        let range = world.ecs.perception_of(pid);
        let discovered = world.ecs.discovered_of(pid);
        for cell in world.get_visible_cells(pos, range) {
            assert!(discovered.contains(&cell));
        }
    }
    
}

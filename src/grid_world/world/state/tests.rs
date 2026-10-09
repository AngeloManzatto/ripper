//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use crate::grid_world::action::Action;
use crate::grid_world::world::state::{World, WorldConfig};
use crate::grid_world::entity::{TileKind};

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_world::{entity::EntityStatus};
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

    fn layout_with_trap()  -> &'static str {
        "#####\n\
         #PTE#\n\
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

    #[test]
    fn step_enemy_hits_trap_and_set_status() 
    {

        // Random generator
        let mut rng = rand::thread_rng();

        // Initialize world
        let mut world = World::from_layout(layout_with_trap(), &WorldConfig::default());
        world.reset();

        // Entity ids
        let player_id = world.ecs.player_id();
        let enemy_id  = world.ecs.enemy_ids()[0];

        // Add actions for each entity
        let mut actions = HashMap::new();
        actions.insert(player_id, Action::Down);
        actions.insert(enemy_id, Action::Left);

        let step =  world.step(actions, &mut rng);

        let enemy_obs = step.observation.entities
            .iter()
            .find(|e| e.id == enemy_id)
            .expect("a trapped enemy must still be in the final observation");

        assert_eq!(enemy_obs.status, EntityStatus::Trapped);


    }

    #[test]
    fn step_enemy_hits_trap_and_is_removed() 
    {

        // Random generator
        let mut rng = rand::thread_rng();

        // Initialize world
        let mut world = World::from_layout(layout_with_trap(), &WorldConfig::default());
        world.reset();

        // Entity ids
        let player_id = world.ecs.player_id();
        let enemy_id  = world.ecs.enemy_ids()[0];

        // Add actions for each entity
        let mut actions = HashMap::new();
        actions.insert(player_id, Action::Down);
        actions.insert(enemy_id, Action::Left);

        world.step(actions, &mut rng);

        assert!(world.ecs.enemy_ids().is_empty());

        let mut actions2 = HashMap::new();
        actions2.insert(player_id, Action::Down);

        let step2 = world.step(actions2, &mut rng);      // player takes any safe action, e.g. Down again or Right
        assert!(step2.observation.entities.iter().all(|e| e.id != enemy_id));

    }

    #[test]
    fn step_player_enemy_hits_trap_at_same_time() 
    {

        // Random generator
        let mut rng = rand::thread_rng();

        // Initialize world
        let mut world = World::from_layout(layout_with_trap(), &WorldConfig::default());
        world.reset();

        // Entity ids
        let player_id = world.ecs.player_id();
        let enemy_id  = world.ecs.enemy_ids()[0];

        // Add actions for each entity
        let mut actions = HashMap::new();
        actions.insert(player_id, Action::Right);
        actions.insert(enemy_id, Action::Left);

        let step = world.step(actions, &mut rng);

        let player_status = world.ecs.status_of(player_id);
        assert_eq!(player_status, EntityStatus::Trapped, "Player Status: {:?}", player_status);

        let enemy_obs = step.observation.entities
            .iter()
            .find(|e| e.id == enemy_id)
            .expect("a trapped enemy must still be in the final observation");

        assert_eq!(enemy_obs.status, EntityStatus::Trapped, "Enemy Status: {:?}", enemy_obs.status);
      
    }

}

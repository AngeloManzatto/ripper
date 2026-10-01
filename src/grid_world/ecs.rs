//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use std::collections::{HashMap, HashSet};

use crate::grid_world::entity::{EntityKind, EntityStatus};

//-----------------------------------------------------
// Entity Component System - ECS
//-----------------------------------------------------

#[derive(Default)]
pub struct Ecs {
    positions: HashMap<u32, (usize, usize)>,
    kinds: HashMap<u32, EntityKind>,
    discovered: HashMap<u32, HashSet<(usize, usize)>>,
    perception_ranges: HashMap<u32, usize>,
    status: HashMap<u32, EntityStatus>,
}

//-----------------------------------------------------
// Entity View
//-----------------------------------------------------
pub struct EntityView {
    pub id: u32,
    pub position: (usize, usize),
    pub kind: EntityKind,
    pub status: EntityStatus,
    pub discovered: HashSet<(usize, usize)>,
    pub perception_range: usize
}

//-----------------------------------------------------
// Spawn / Despawn
//-----------------------------------------------------

impl Ecs {

    // Add a new entity
    pub fn spawn(&mut self, id: u32, kind: EntityKind, pos: (usize, usize), perception_range: usize) {
        self.positions.insert(id, pos);
        self.kinds.insert(id, kind);
        self.discovered.insert(id, HashSet::new());
        self.perception_ranges.insert(id, perception_range);
        self.status.insert(id, EntityStatus::Alive);
    }

    // Remove entity
    pub fn despawn(&mut self, id: u32) {
        self.positions.remove(&id).unwrap_or_else(|| panic!("ID '{}' not found in positions", id));
        self.kinds.remove(&id).unwrap_or_else(|| panic!("ID '{}' not found in kinds", id));
        self.discovered.remove(&id).unwrap_or_else(|| panic!("ID '{}' not found in discovered", id));
        self.perception_ranges.remove(&id).unwrap_or_else(|| panic!("ID '{}' not found in perception_ranges", id));
        self.status.remove(&id).unwrap_or_else(|| panic!("ID '{}' not found in status", id));
    }

}

//-----------------------------------------------------
// Acessors
//-----------------------------------------------------

impl Ecs {

    // All currently-alive entity ids.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.positions.keys().copied()
    }

    // Ids matching a given kind — e.g. all enemies, without any
    pub fn ids_of_kind(&self, kind: EntityKind) -> impl Iterator<Item = u32> + '_ {
        self.kinds.iter().filter(move |(_, k)| **k == kind).map(|(&id, _)| id)
    }
}

impl Ecs {

    pub fn position_of(&self, id: u32) -> (usize, usize) {
        self.positions.get(&id).copied().unwrap_or_else(|| panic!("ID '{}' not found in position", id))
    }

    pub fn discovered_of(&self, id: u32) -> &HashSet<(usize, usize)> {
        self.discovered.get(&id).unwrap_or_else(|| panic!("ID '{}' not found in discovered", id))
    }

    pub fn kind_of(&self, id: u32) -> EntityKind {
        self.kinds.get(&id).copied().unwrap_or_else(|| panic!("ID '{}' not found in kind", id))
    }

    pub fn perception_of(&self, id: u32) -> usize {
        self.perception_ranges.get(&id).copied().unwrap_or_else(|| panic!("ID '{}' not found in perception", id))
    }

    pub fn status_of(&self, id: u32) -> EntityStatus {
        self.status.get(&id).copied().unwrap_or(EntityStatus::Alive)
    }

}

impl Ecs {

    // Get player ID
    pub fn player_id(&self) -> u32 {
        self.ids_of_kind(EntityKind::Player)
            .next()
            .expect("no Player entity found")
    }

    // Get enemy IDs
    pub fn enemy_ids(&self) -> Vec<u32> {
        self.ids_of_kind(EntityKind::Enemy).collect()
    }
}

impl Ecs {

    // Overwrites an entity's position. The only sanctioned way to move
    pub fn set_position(&mut self, id: u32, pos: (usize, usize)) {
        *self.positions.get_mut(&id).unwrap_or_else(|| panic!("ID '{}' not found in positions", id)) = pos;
    }

    // Overwrites an entity's status (Trapped/Caught/GoalReached/...).
    pub fn set_status(&mut self, id: u32, status: EntityStatus) {
        *self.status.get_mut(&id).unwrap_or_else(|| panic!("ID '{}' not found in status", id)) = status;
    }

    // Adds a tile to an entity's discovered set — the FOW update each tick.
    pub fn set_discovered(&mut self, id: u32, pos: (usize, usize)) {
        self.discovered.get_mut(&id).unwrap_or_else(|| panic!("ID '{}' not found in discovered", id)).insert(pos);
    }
}


impl Ecs {

    // Return entity view
    pub fn view(&self, id: u32) -> EntityView {
        EntityView {
            id,
            position: self.position_of(id),
            kind: self.kind_of(id),
            status: self.status_of(id),
            perception_range : self.perception_of(id),
            discovered: self.discovered_of(id).clone()
        }
    }

    /// Everything about the player in one call.
    pub fn player(&self) -> EntityView {
        self.view(self.player_id())
    }

    /// Everything about every enemy in one call.
    pub fn enemies(&self) -> Vec<EntityView> {
        self.enemy_ids().into_iter().map(|id| self.view(id)).collect()
    }
}

//-----------------------------------------------------
// Validation
//-----------------------------------------------------

impl Ecs {
    #[cfg(debug_assertions)]
    pub fn assert_consistent(&self) {
        let ids: std::collections::HashSet<_> = self.positions.keys().copied().collect();
        for (name, keys) in [
            ("kinds", self.kinds.keys().copied().collect::<std::collections::HashSet<_>>()),
            ("discovered", self.discovered.keys().copied().collect()),
            ("perception_ranges", self.perception_ranges.keys().copied().collect()),
            ("status", self.status.keys().copied().collect()),
        ] {
            assert_eq!(ids, keys, "Ecs component maps out of sync: 'positions' vs '{}'", name);
        }
    }
}

//-----------------------------------------------------
// Debug
//-----------------------------------------------------

impl std::fmt::Debug for Ecs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

        writeln!(f, "{:<6} {:<10} {:<8} {:<12} {:<10}", "id", "position", "kind", "status", "perception")?;

        let mut ids: Vec<_> = self.positions.keys().copied().collect();
        ids.sort();
        for id in ids {
            writeln!(f, "{:<6} {:<10?} {:<8?} {:<12?} {:<10?}",
                id,
                self.positions.get(&id),
                self.kinds.get(&id),
                self.status.get(&id),
                self.perception_ranges.get(&id),
            )?;
        }
        Ok(())
    }
}

//-----------------------------------------------------
// Test
//-----------------------------------------------------

#[test]
fn despawn_removes_entity_from_every_component_map() {
    let mut ecs = Ecs::default();
    ecs.spawn(1, EntityKind::Enemy, (0, 0), 3);
    ecs.despawn(1);
    ecs.assert_consistent(); // trivially true here since ecs is empty, but exercises the check
    assert!(ecs.positions.is_empty());
    assert!(ecs.status.is_empty()); // this line alone would have caught the earlier bug
}
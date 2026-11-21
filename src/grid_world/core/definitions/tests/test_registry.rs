//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::grid_world::core::definitions::registry::{Registry, TerrainDef, TerrainId};

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

fn void() -> TerrainDef {
    TerrainDef {
        id: String::from("base:void"),
        glyph: None,
        walkable: false,
        transparent: false,
    }
}

fn empty() -> TerrainDef {
    TerrainDef { id: String::from("base:empty"), glyph: Some('.'), walkable: true, transparent: true }
}

fn wall() -> TerrainDef {
    TerrainDef { id: String::from("base:wall"), glyph: Some('#'), walkable: false, transparent: false }
}

#[test]
fn a_terrain_id_is_copied_not_moved() {
    let first = TerrainId(1);
    let second = first;

    assert_eq!(first, second);
}

#[test]
fn terrain_ids_are_ordered_by_their_number() {
    assert!(TerrainId(0) < TerrainId(1));
    assert_eq!(TerrainId(7), TerrainId(7));
}

#[test]
fn a_cloned_definition_is_equal_to_the_original() {
    let wall = TerrainDef {
        id: String::from("base:wall"),
        glyph: Some('#'),
        walkable: false,
        transparent: false,
    };

    let copy = wall.clone();

    assert_eq!(wall, copy);
}



#[test]
fn a_registry_counts_its_terrains() {
    let registry = Registry::new(vec![empty(), wall()]);

    assert_eq!(registry.terrain_count(), 2);
}

#[test]
fn a_terrain_is_found_by_its_position_in_the_list() {
    let registry = Registry::new(vec![empty(), wall()]);

    assert_eq!(registry.terrain(TerrainId(0)).id, "base:empty");
    assert_eq!(registry.terrain(TerrainId(1)).id, "base:wall");
}

#[test]
fn a_terrain_is_found_by_its_name() {
    let registry = Registry::new(vec![empty(), wall()]);

    assert_eq!(registry.terrain_id("base:empty"), Some(TerrainId(0)));
    assert_eq!(registry.terrain_id("base:wall"), Some(TerrainId(1)));
}

#[test]
fn an_unknown_name_gives_nothing() {
    let registry = Registry::new(vec![empty(), wall()]);

    assert_eq!(registry.terrain_id("base:lava"), None);
}

#[test]
fn the_id_found_by_name_leads_back_to_the_same_terrain() {
    let registry = Registry::new(vec![empty(), wall()]);

    let id = registry.terrain_id("base:wall").unwrap();

    assert_eq!(registry.terrain(id).id, "base:wall");
}

#[test]
fn walkable_follows_the_definition() {
    let registry = Registry::new(vec![empty(), wall()]);

    assert!(registry.walkable(TerrainId(0)));
    assert!(!registry.walkable(TerrainId(1)));
}

#[test]
fn transparent_follows_the_definition() {
    let registry = Registry::new(vec![empty(), wall()]);

    assert!(registry.transparent(TerrainId(0)));
    assert!(!registry.transparent(TerrainId(1)));
}

#[test]
fn the_answers_follow_the_id_not_the_name() {
    let registry = Registry::new(vec![wall(), empty()]);

    assert!(!registry.walkable(TerrainId(0)));
    assert!(registry.walkable(TerrainId(1)));
}

#[test]
fn a_terrain_is_found_by_its_glyph() {
    let registry = Registry::new(vec![empty(), wall()]);

    assert_eq!(registry.terrain_id_by_glyph('.'), Some(TerrainId(0)));
    assert_eq!(registry.terrain_id_by_glyph('#'), Some(TerrainId(1)));
}

#[test]
fn an_unknown_glyph_gives_nothing() {
    let registry = Registry::new(vec![empty(), wall()]);

    assert_eq!(registry.terrain_id_by_glyph('?'), None);
}

#[test]
fn a_terrain_without_a_glyph_cannot_be_found_by_glyph_but_still_has_its_number() {
    let registry = Registry::new(vec![empty(), void(), wall()]);

    assert_eq!(registry.terrain_id_by_glyph('#'), Some(TerrainId(2)));
    assert_eq!(registry.terrain_id("base:void"), Some(TerrainId(1)));
    assert_eq!(registry.glyph(TerrainId(1)), None);
}

#[test]
fn the_glyph_of_an_id_leads_back_to_the_same_id() {
    let registry = Registry::new(vec![empty(), wall()]);

    let glyph = registry.glyph(TerrainId(1)).unwrap();

    assert_eq!(glyph, '#');
    assert_eq!(registry.terrain_id_by_glyph(glyph), Some(TerrainId(1)));
}
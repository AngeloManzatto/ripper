//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::super::store::{Content, TerrainDef, TerrainId};

//-----------------------------------------------------
// Helpers
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

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

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
fn a_content_counts_its_terrains() {
    let content = Content::new(vec![empty(), wall()]);

    assert_eq!(content.terrains().len(), 2);
}

#[test]
fn a_terrain_is_found_by_its_position_in_the_list() {
    let content = Content::new(vec![empty(), wall()]);

    assert_eq!(content.terrain(TerrainId(0)).id, "base:empty");
    assert_eq!(content.terrain(TerrainId(1)).id, "base:wall");
}

#[test]
fn a_terrain_is_found_by_its_name() {
    let content = Content::new(vec![empty(), wall()]);

    assert_eq!(content.terrain_id("base:empty"), Some(TerrainId(0)));
    assert_eq!(content.terrain_id("base:wall"), Some(TerrainId(1)));
}

#[test]
fn an_unknown_name_gives_nothing() {
    let content = Content::new(vec![empty(), wall()]);

    assert_eq!(content.terrain_id("base:lava"), None);
}

#[test]
fn the_id_found_by_name_leads_back_to_the_same_terrain() {
    let content = Content::new(vec![empty(), wall()]);

    let id = content.terrain_id("base:wall").unwrap();

    assert_eq!(content.terrain(id).id, "base:wall");
}

#[test]
fn walkable_follows_the_definition() {
    let content = Content::new(vec![empty(), wall()]);

    assert!(content.terrain(TerrainId(0)).walkable);
    assert!(!content.terrain(TerrainId(1)).walkable);
}

#[test]
fn transparent_follows_the_definition() {
    let content = Content::new(vec![empty(), wall()]);

    assert!(content.terrain(TerrainId(0)).transparent);
    assert!(!content.terrain(TerrainId(1)).transparent);
}

#[test]
fn the_answers_follow_the_id_not_the_name() {
    let content = Content::new(vec![wall(), empty()]);

    assert!(!content.terrain(TerrainId(0)).walkable);
    assert!(content.terrain(TerrainId(1)).walkable);
}

#[test]
fn a_terrain_is_found_by_its_glyph() {
    let content = Content::new(vec![empty(), wall()]);

    assert_eq!(content.terrain_id_by_glyph('.'), Some(TerrainId(0)));
    assert_eq!(content.terrain_id_by_glyph('#'), Some(TerrainId(1)));
}

#[test]
fn an_unknown_glyph_gives_nothing() {
    let content = Content::new(vec![empty(), wall()]);

    assert_eq!(content.terrain_id_by_glyph('?'), None);
}

#[test]
fn a_terrain_without_a_glyph_cannot_be_found_by_glyph_but_still_has_its_number() {
    let content = Content::new(vec![empty(), void(), wall()]);

    assert_eq!(content.terrain_id_by_glyph('#'), Some(TerrainId(2)));
    assert_eq!(content.terrain_id("base:void"), Some(TerrainId(1)));
    assert_eq!(content.terrain(TerrainId(1)).glyph, None);
}

#[test]
fn the_glyph_of_an_id_leads_back_to_the_same_id() {
    let content = Content::new(vec![empty(), wall()]);

    let glyph = content.terrain(TerrainId(1)).glyph.unwrap();

    assert_eq!(glyph, '#');
    assert_eq!(content.terrain_id_by_glyph(glyph), Some(TerrainId(1)));
}

#[test]
fn the_table_of_terrains_can_be_asked_by_name() {
    let content = Content::new(vec![empty(), wall()]);

    let found = content.terrains().get_by_name("base:wall").unwrap();

    assert_eq!(found.glyph, Some('#'));
}
//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::super::store::{Content, TerrainId};

use crate::ripper::test_support::{
    EMPTY_ID, WALL_ID, base_content, empty, void, wall
};


//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn a_terrain_id_is_copied_not_moved() {
    let first = WALL_ID;
    let second = first;

    assert_eq!(first, second);
}

#[test]
fn terrain_ids_are_ordered_by_their_number() {
    assert!(EMPTY_ID < WALL_ID);
    assert_eq!(TerrainId(7), TerrainId(7));
}

#[test]
fn a_cloned_definition_is_equal_to_the_original() {
    let wall = wall();

    let copy = wall.clone();

    assert_eq!(wall, copy);
}



#[test]
fn a_content_counts_its_terrains() {
    let content = base_content();

    assert_eq!(content.terrains().len(), 2);
}

#[test]
fn a_terrain_is_found_by_its_position_in_the_list() {
    let content = base_content();

    assert_eq!(content.terrain(EMPTY_ID).id, "base:empty");
    assert_eq!(content.terrain(WALL_ID).id, "base:wall");
}

#[test]
fn a_terrain_is_found_by_its_name() {
    let content = base_content();

    assert_eq!(content.terrain_id("base:empty"), Some(EMPTY_ID));
    assert_eq!(content.terrain_id("base:wall"), Some(WALL_ID));
}

#[test]
fn an_unknown_name_gives_nothing() {
    let content = base_content();

    assert_eq!(content.terrain_id("base:lava"), None);
}

#[test]
fn the_id_found_by_name_leads_back_to_the_same_terrain() {
    let content = base_content();

    let id = content.terrain_id("base:wall").unwrap();

    assert_eq!(content.terrain(id).id, "base:wall");
}

#[test]
fn walkable_follows_the_definition() {
    let content = base_content();

    assert!(content.terrain(EMPTY_ID).walkable);
    assert!(!content.terrain(WALL_ID).walkable);
}

#[test]
fn transparent_follows_the_definition() {
    let content = base_content();

    assert!(content.terrain(EMPTY_ID).transparent);
    assert!(!content.terrain(WALL_ID).transparent);
}

#[test]
fn the_answers_follow_the_id_not_the_name() {
    let content = base_content();

    assert!(!content.terrain(WALL_ID).walkable);
    assert!(content.terrain(EMPTY_ID).walkable);
}

#[test]
fn a_terrain_is_found_by_its_glyph() {
    let content = base_content();

    assert_eq!(content.terrain_id_by_glyph('.'), Some(EMPTY_ID));
    assert_eq!(content.terrain_id_by_glyph('#'), Some(WALL_ID));
}

#[test]
fn an_unknown_glyph_gives_nothing() {
    let content = base_content();

    assert_eq!(content.terrain_id_by_glyph('?'), None);
}

#[test]
fn a_terrain_without_a_glyph_cannot_be_found_by_glyph_but_still_has_its_number() {
    let content = Content::new(vec![empty(), void(), wall()]);

    assert_eq!(content.terrain_id_by_glyph('#'), Some(TerrainId(2)));
    assert_eq!(content.terrain_id("base:void"), Some(WALL_ID));
    assert_eq!(content.terrain(WALL_ID).glyph, None);
}

#[test]
fn the_glyph_of_an_id_leads_back_to_the_same_id() {
    let content = base_content();

    let glyph = content.terrain(WALL_ID).glyph.unwrap();

    assert_eq!(glyph, '#');
    assert_eq!(content.terrain_id_by_glyph(glyph), Some(WALL_ID));
}

#[test]
fn the_table_of_terrains_can_be_asked_by_name() {
    let content = base_content();

    let found = content.terrains().get_by_name("base:wall").unwrap();

    assert_eq!(found.glyph, Some('#'));
}
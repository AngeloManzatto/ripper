//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::fixture::{pack_with, plain_object};
use super::super::build::build_terrains;

use crate::ripper::content::store::TerrainDef;

//-----------------------------------------------------
// Helpers
//-----------------------------------------------------

fn ids(terrains: &[TerrainDef]) -> Vec<String> {
    let mut ids = Vec::new();
    for terrain in terrains {
        ids.push(terrain.id.clone());
    }
    ids
}

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn terrains_follow_the_load_order_then_the_id() {
    let packs = vec![
        pack_with("ice", vec![plain_object("ice:snow", Some("*"))]),
        pack_with("base", vec![plain_object("base:wall", Some("#")), plain_object("base:empty", Some("."))]),
    ];

    let terrains = build_terrains(&packs, &[1, 0]);

    assert_eq!(ids(&terrains), vec!["base:empty", "base:wall", "ice:snow"]);
}

#[test]
fn the_glyph_becomes_a_character_and_a_missing_glyph_stays_missing() {
    let packs = vec![pack_with("base", vec![plain_object("base:empty", None), plain_object("base:wall", Some("#"))])];

    let terrains = build_terrains(&packs, &[0]);

    assert_eq!(terrains[0].glyph, None);
    assert_eq!(terrains[1].glyph, Some('#'));
}

#[test]
fn the_flags_are_copied() {
    let mut solid = plain_object("base:wall", Some("#"));
    solid.terrain.walkable = false;
    solid.terrain.transparent = false;
    let packs = vec![pack_with("base", vec![solid])];

    let terrains = build_terrains(&packs, &[0]);

    assert!(!terrains[0].walkable);
    assert!(!terrains[0].transparent);
}

#[test]
fn no_packs_give_no_terrains() {
    assert!(build_terrains(&[], &[]).is_empty());
}

#[test]
fn the_order_of_the_files_does_not_change_the_result() {
    let one = vec![pack_with("base", vec![plain_object("base:a", None), plain_object("base:b", None)])];
    let other = vec![pack_with("base", vec![plain_object("base:b", None), plain_object("base:a", None)])];

    assert_eq!(build_terrains(&one, &[0]), build_terrains(&other, &[0]));
}

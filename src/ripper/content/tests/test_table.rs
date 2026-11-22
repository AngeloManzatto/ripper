//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::super::store::TerrainDef;
use super::super::table::Named;
use super::super::table::Table;

//-----------------------------------------------------
// Helpers
//-----------------------------------------------------

struct Thing {
    name: String,
}

impl Named for Thing {
    fn name(&self) -> &str {
        &self.name
    }
}

fn things(names: &[&str]) -> Table<Thing> {
    let defs = names.iter().map(|n| Thing { name: n.to_string() }).collect();
    Table::new(defs)
}

//-----------------------------------------------------
// Test
//-----------------------------------------------------

#[test]
fn a_terrain_is_named_by_its_id() {
    let wall = TerrainDef {
        id: "base:wall".to_string(),
        glyph: Some('#'),
        walkable: false,
        transparent: false,
    };

    assert_eq!(wall.name(), "base:wall");

}

#[test]
fn a_table_knows_how_many_definitions_it_holds() {
    assert_eq!(things(&["a", "b", "c"]).len(), 3);
}

#[test]
fn a_table_with_nothing_in_it_is_empty() {
    assert!(things(&[]).is_empty());
    assert!(!things(&["a"]).is_empty());
}

#[test]
fn a_name_leads_to_its_position() {
    let table = things(&["a", "b", "c"]);
    assert_eq!(table.position_of("a"), Some(0));
    assert_eq!(table.position_of("c"), Some(2));
}

#[test]
fn an_unknown_name_gives_nothing() {
    assert_eq!(things(&["a"]).position_of("zzz"), None);
}

#[test]
fn a_position_leads_back_to_its_definition() {
    let table = things(&["a", "b"]);
    assert_eq!(table.get(1).unwrap().name(), "b");
}

#[test]
fn a_position_past_the_end_gives_nothing() {
    assert!(things(&["a", "b"]).get(2).is_none());
}

#[test]
fn the_name_of_a_position_leads_back_to_the_same_position() {
    let table = things(&["x", "y", "z"]);
    let position = table.position_of("y").unwrap();
    assert_eq!(table.get(position).unwrap().name(), "y");
}

#[test]
fn a_definition_is_found_directly_by_its_name() {
    let table = things(&["a", "b"]);

    assert_eq!(table.get_by_name("b").unwrap().name(), "b");
}

#[test]
fn an_unknown_name_gives_no_definition() {
    assert!(things(&["a"]).get_by_name("zzz").is_none());
}
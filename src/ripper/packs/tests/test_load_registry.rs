//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::fs;
use std::path::PathBuf;

use super::super::load_content::load_content;

use crate::ripper::content::store::TerrainId;

//-----------------------------------------------------
// Constants
//-----------------------------------------------------

const WALL: &str = r##"{"category":"terrain","id":"base:wall","glyph":"#","walkable":false,"transparent":false}"##;
const EMPTY: &str = r##"{"category":"terrain","id":"base:empty","glyph":".","walkable":true,"transparent":true}"##;
const BASE: &str = r#"{"name":"base","version":"0.1.0"}"#;

//-----------------------------------------------------
// Helpers
//-----------------------------------------------------

fn fresh_temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_pack(root: &PathBuf, folder: &str, pack_json: &str, objects: &[(&str, &str)]) {
    let dir = root.join(folder);
    fs::create_dir_all(dir.join("terrain")).unwrap();
    fs::write(dir.join("pack.json"), pack_json).unwrap();
    for (file, content) in objects {
        fs::write(dir.join("terrain").join(file), content).unwrap();
    }
}

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn a_good_folder_gives_a_registry_with_numbered_terrains() {
    let root = fresh_temp_dir("ripper_registry_ok");
    write_pack(&root, "base", BASE, &[("wall.json", WALL), ("empty.json", EMPTY)]);

    let registry = load_content(&root).unwrap();

    assert_eq!(registry.terrains().len(), 2);
    assert_eq!(registry.terrain_id("base:empty"), Some(TerrainId(0)));
    assert_eq!(registry.terrain_id("base:wall"), Some(TerrainId(1)));
}

#[test]
fn a_pack_that_depends_on_another_is_numbered_after_it() {
    let root = fresh_temp_dir("ripper_registry_two_packs");
    let ice = r#"{"name":"ice","version":"0.1.0","depends_on":["base"]}"#;
    let snow = r##"{"category":"terrain","id":"ice:snow","glyph":"*","walkable":true,"transparent":true}"##;
    write_pack(&root, "ice", ice, &[("snow.json", snow)]);
    write_pack(&root, "base", BASE, &[("wall.json", WALL)]);

    let registry = load_content(&root).unwrap();

    assert_eq!(registry.terrain_id("base:wall"), Some(TerrainId(0)));
    assert_eq!(registry.terrain_id("ice:snow"), Some(TerrainId(1)));
}

#[test]
fn a_file_that_cannot_be_read_stops_the_load() {
    let root = fresh_temp_dir("ripper_registry_unreadable");
    write_pack(&root, "base", BASE, &[("wall.json", WALL), ("empty.json", "not json")]);

    assert!(load_content(&root).is_err());
}

#[test]
fn a_broken_rule_stops_the_load_and_every_rule_problem_is_reported() {
    let root = fresh_temp_dir("ripper_registry_rules");
    write_pack(&root, "base", BASE, &[("wall.json", WALL), ("wall_copy.json", WALL)]);

    let errors = load_content(&root).unwrap_err();

    assert_eq!(errors.len(), 3);
}

#[test]
fn a_missing_root_is_an_error() {
    let root = std::env::temp_dir().join("ripper_registry_missing");
    let _ = fs::remove_dir_all(&root);

    assert_eq!(load_content(&root).unwrap_err().len(), 1);
}
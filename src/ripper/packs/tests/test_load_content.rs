//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::fs;
use std::path::{Path, PathBuf};

use super::super::load_content::load_content;
use super::fixture::{BASE, WALL, EMPTY, ICE, SNOW};

use crate::ripper::content::store::TerrainId;

//-----------------------------------------------------
// Helpers
//-----------------------------------------------------

pub fn fresh_temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn write_pack(root: &Path, folder: &str, pack_json: &str, objects: &[(&str, &str)]) {
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
fn a_good_folder_gives_a_content_with_numbered_terrains() {
    let root = fresh_temp_dir("ripper_content_ok");
    write_pack(&root, "base", BASE, &[("wall.json", WALL), ("empty.json", EMPTY)]);

    let content = load_content(&root).unwrap();

    assert_eq!(content.terrains().len(), 2);
    assert_eq!(content.terrain_id("base:empty"), Some(TerrainId(0)));
    assert_eq!(content.terrain_id("base:wall"), Some(TerrainId(1)));
}

#[test]
fn a_pack_that_depends_on_another_is_numbered_after_it() {
    let root = fresh_temp_dir("ripper_content_two_packs");

    
    write_pack(&root, "ice", ICE, &[("snow.json", SNOW)]);
    write_pack(&root, "base", BASE, &[("wall.json", WALL)]);

    let content = load_content(&root).unwrap();

    assert_eq!(content.terrain_id("base:wall"), Some(TerrainId(0)));
    assert_eq!(content.terrain_id("ice:snow"), Some(TerrainId(1)));
}

#[test]
fn a_file_that_cannot_be_read_stops_the_load() {
    let root = fresh_temp_dir("ripper_content_unreadable");
    write_pack(&root, "base", BASE, &[("wall.json", WALL), ("empty.json", "not json")]);

    assert!(load_content(&root).is_err());
}

#[test]
fn a_broken_rule_stops_the_load_and_every_rule_problem_is_reported() {
    let root = fresh_temp_dir("ripper_content_rules");
    write_pack(&root, "base", BASE, &[("wall.json", WALL), ("wall_copy.json", WALL)]);

    let errors = load_content(&root).unwrap_err();

    assert_eq!(errors.len(), 3);
}

#[test]
fn a_missing_root_is_an_error() {
    let root = std::env::temp_dir().join("ripper_content_missing");
    let _ = fs::remove_dir_all(&root);

    assert_eq!(load_content(&root).unwrap_err().len(), 1);
}

#[test]
fn adding_a_pack_that_depends_on_base_does_not_change_the_numbers_of_base() {

    let alone = fresh_temp_dir("ripper_ids_alone");
    write_pack(&alone, "base", BASE, &[("wall.json", WALL), ("empty.json", EMPTY)]);
    let before = load_content(&alone).unwrap();

    let together = fresh_temp_dir("ripper_ids_together");
    write_pack(&together, "base", BASE, &[("wall.json", WALL), ("empty.json", EMPTY)]);
    write_pack(&together, "ice", ICE, &[("snow.json", SNOW)]);
    let after = load_content(&together).unwrap();

    for name in ["base:empty", "base:wall"] {
        assert_eq!(before.terrain_id(name), after.terrain_id(name));
    }
}
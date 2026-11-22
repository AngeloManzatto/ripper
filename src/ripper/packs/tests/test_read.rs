//-----------------------------------------------------
// Tests
//-----------------------------------------------------

use super::super::read::{
    read_object, 
    read_pack, 
    sorted_entries, 
    collect_json_files,
    find_json_files,
    load_pack,
    load_all_packs
};
use super::super::raw::RawObject;
use std::fs;
use std::path::PathBuf;

//-----------------------------------------------------
// Constants
//-----------------------------------------------------

const WALL : &str = r##"{"category":"terrain","id":"base:wall","glyph":"#","walkable":false,"transparent":false}"##;
const EMPTY: &str = r##"{"category":"terrain","id":"base:empty","glyph":".","walkable":true,"transparent":true}"##;
const PACK : &str = r#"{"name":"base","version":"0.1.0"}"#;

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

fn fresh_temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_temp(name: &str, content: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    fs::write(&path, content).unwrap();
    path
}

#[test]
fn a_valid_file_is_read() {
    let path = write_temp(
        "ripper_read_valid.json",
        r##"{"category":"terrain","id":"base:wall","glyph":"#","walkable":false,"transparent":false}"##,
    );

    let object = read_object(&path).unwrap();

    match object {
        RawObject::Terrain(terrain) => assert_eq!(terrain.id, "base:wall"),
    }
}

#[test]
fn a_missing_file_is_an_error_that_names_the_file() {
    let path = std::env::temp_dir().join("ripper_read_does_not_exist.json");

    let error = read_object(&path).unwrap_err();

    assert_eq!(error.path, path);
    assert!(error.message.contains("cannot read"));
}

#[test]
fn text_that_is_not_json_is_an_error() {
    let path = write_temp("ripper_read_not_json.json", "this is not json");

    let error = read_object(&path).unwrap_err();

    assert!(error.message.contains("invalid definition"));
}

#[test]
fn an_unknown_field_is_an_error() {
    let path = write_temp(
        "ripper_read_unknown_field.json",
        r##"{"category":"terrain","id":"base:wall","walkable":false,"transparent":false,"colour":"red"}"##,
    );

    let error = read_object(&path).unwrap_err();

    assert!(error.message.contains("colour"));
}

#[test]
fn entries_come_back_sorted_by_name() {
    let dir = fresh_temp_dir("ripper_sorted_entries");
    fs::write(dir.join("c.json"), "").unwrap();
    fs::write(dir.join("a.json"), "").unwrap();
    fs::write(dir.join("b.json"), "").unwrap();

    let paths = sorted_entries(&dir).unwrap();

    assert_eq!(paths, vec![dir.join("a.json"), dir.join("b.json"), dir.join("c.json")]);
}

#[test]
fn a_missing_folder_is_an_error_that_names_the_folder() {
    let dir = std::env::temp_dir().join("ripper_folder_that_does_not_exist");
    let _ = fs::remove_dir_all(&dir);

    let error = sorted_entries(&dir).unwrap_err();

    assert_eq!(error.path, dir);
    assert!(error.message.contains("cannot read the folder"));
}


#[test]
fn a_valid_pack_file_is_read() {
    let path = write_temp(
        "ripper_read_pack_valid.json",
        r#"{"name":"base","version":"0.1.0","depends_on":[]}"#,
    );

    let pack = read_pack(&path).unwrap();

    assert_eq!(pack.name, "base");
    assert_eq!(pack.version, "0.1.0");
}

#[test]
fn a_pack_file_without_a_name_is_an_error() {
    let path = write_temp("ripper_read_pack_no_name.json", r#"{"version":"0.1.0"}"#);

    let error = read_pack(&path).unwrap_err();

    assert!(error.message.contains("name"));
}

#[test]
fn json_files_are_found_in_subfolders_in_a_fixed_order() {
    let dir = fresh_temp_dir("ripper_collect_json");
    fs::create_dir_all(dir.join("terrain").join("sub")).unwrap();
    fs::write(dir.join("pack.json"), "{}").unwrap();
    fs::write(dir.join("notes.txt"), "").unwrap();
    fs::write(dir.join("terrain").join("wall.json"), "{}").unwrap();
    fs::write(dir.join("terrain").join("empty.json"), "{}").unwrap();
    fs::write(dir.join("terrain").join("sub").join("deep.json"), "{}").unwrap();
    let mut found = Vec::new();
    let mut errors = Vec::new();

    collect_json_files(&dir, &mut found, &mut errors);

    assert!(errors.is_empty());
    assert_eq!(
        found,
        vec![
            dir.join("pack.json"),
            dir.join("terrain").join("empty.json"),
            dir.join("terrain").join("sub").join("deep.json"),
            dir.join("terrain").join("wall.json"),
        ]
    );
}

#[test]
fn a_folder_that_cannot_be_read_is_reported_not_hidden() {
    let dir = std::env::temp_dir().join("ripper_collect_missing");
    let _ = fs::remove_dir_all(&dir);
    let mut found = Vec::new();
    let mut errors = Vec::new();

    collect_json_files(&dir, &mut found, &mut errors);

    assert!(found.is_empty());
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, dir);
}

#[test]
fn find_json_files_returns_the_files_in_a_fixed_order() {
    let dir = fresh_temp_dir("ripper_find_json");
    fs::create_dir_all(dir.join("terrain")).unwrap();
    fs::write(dir.join("pack.json"), "{}").unwrap();
    fs::write(dir.join("notes.txt"), "").unwrap();
    fs::write(dir.join("terrain").join("wall.json"), "{}").unwrap();
    fs::write(dir.join("terrain").join("empty.json"), "{}").unwrap();

    let found = find_json_files(&dir).unwrap();

    assert_eq!(
        found,
        vec![
            dir.join("pack.json"),
            dir.join("terrain").join("empty.json"),
            dir.join("terrain").join("wall.json"),
        ]
    );
}

#[test]
fn find_json_files_in_an_empty_folder_gives_an_empty_list() {
    let dir = fresh_temp_dir("ripper_find_json_empty");

    let found = find_json_files(&dir).unwrap();

    assert!(found.is_empty());
}

#[test]
fn find_json_files_reports_a_missing_folder() {
    let dir = std::env::temp_dir().join("ripper_find_json_missing");
    let _ = fs::remove_dir_all(&dir);

    let errors = find_json_files(&dir).unwrap_err();

    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, dir);
}

#[test]
fn a_pack_folder_is_loaded_with_its_objects() {
    let dir = fresh_temp_dir("ripper_load_pack_ok");
    fs::create_dir_all(dir.join("terrain")).unwrap();
    fs::write(dir.join("pack.json"), PACK).unwrap();
    fs::write(dir.join("terrain").join("wall.json"), WALL).unwrap();
    fs::write(dir.join("terrain").join("empty.json"), EMPTY).unwrap();

    let pack = load_pack(&dir).unwrap();

    assert_eq!(pack.raw.name, "base");
    assert_eq!(pack.objects.len(), 2);
    assert_eq!(pack.objects[0].terrain.id, "base:empty");
    assert_eq!(pack.objects[1].terrain.id, "base:wall");
}

#[test]
fn a_pack_folder_without_pack_json_is_an_error() {
    let dir = fresh_temp_dir("ripper_load_pack_no_manifest");

    let errors = load_pack(&dir).unwrap_err();

    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, dir);
    assert!(errors[0].message.contains("pack.json"));
}

#[test]
fn every_broken_file_is_reported_not_just_the_first() {
    let dir = fresh_temp_dir("ripper_load_pack_many_errors");
    fs::write(dir.join("pack.json"), r#"{"version":"0.1.0"}"#).unwrap();
    fs::write(dir.join("a.json"), "not json").unwrap();
    fs::write(dir.join("b.json"), WALL).unwrap();
    fs::write(dir.join("c.json"), "{}").unwrap();

    let errors = load_pack(&dir).unwrap_err();

    assert_eq!(errors.len(), 3);
}

//

fn write_pack(root: &PathBuf, folder: &str, pack_json: &str, objects: &[(&str, &str)]) {
    let dir = root.join(folder);
    fs::create_dir_all(dir.join("terrain")).unwrap();
    fs::write(dir.join("pack.json"), pack_json).unwrap();
    for (file, content) in objects {
        fs::write(dir.join("terrain").join(file), content).unwrap();
    }
}

#[test]
fn every_pack_folder_in_the_root_is_loaded_in_folder_order() {
    let root = fresh_temp_dir("ripper_load_all_ok");
    write_pack(&root, "ice", r#"{"name":"ice","version":"0.1.0"}"#, &[]);
    write_pack(&root, "base", PACK, &[("wall.json", WALL)]);
    fs::write(root.join("README.txt"), "not a pack").unwrap();

    let packs = load_all_packs(&root).unwrap();

    assert_eq!(packs.len(), 2);
    assert_eq!(packs[0].raw.name, "base");
    assert_eq!(packs[0].objects.len(), 1);
    assert_eq!(packs[1].raw.name, "ice");
}

#[test]
fn an_empty_root_gives_no_packs() {
    let root = fresh_temp_dir("ripper_load_all_empty");

    let packs = load_all_packs(&root).unwrap();

    assert!(packs.is_empty());
}

#[test]
fn a_missing_root_is_one_error() {
    let root = std::env::temp_dir().join("ripper_load_all_missing");
    let _ = fs::remove_dir_all(&root);

    let errors = load_all_packs(&root).unwrap_err();

    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, root);
}

#[test]
fn broken_packs_are_all_reported_and_no_packs_are_returned() {
    let root = fresh_temp_dir("ripper_load_all_broken");
    write_pack(&root, "base", PACK, &[("wall.json", WALL)]);
    write_pack(&root, "ice", "not json", &[]);
    write_pack(&root, "fire", PACK, &[("bad.json", "{}")]);

    let errors = load_all_packs(&root).unwrap_err();

    assert_eq!(errors.len(), 2);
}

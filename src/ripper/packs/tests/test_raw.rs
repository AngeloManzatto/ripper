//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::ripper::packs::raw::{RawPack, RawTerrain, RawObject};

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn reads_a_complete_terrain_file() {

    let text = r##"
    { 
        "format_version": 1, 
        "id": "base:wall", 
        "description": "Solid.", 
        "glyph": "#", 
        "walkable": false, 
        "transparent": false 
    }"##;

    let terrain: RawTerrain = serde_json::from_str(text).unwrap();

    assert_eq!(terrain.id, "base:wall");
    assert_eq!(terrain.glyph, Some(String::from("#")));
    assert_eq!(terrain.walkable, false);
    assert_eq!(terrain.format_version, Some(1));
}

#[test]
fn optional_fields_can_be_left_out() {
    let text = r#"
    { 
        "id": "base:void", 
        "walkable": false, 
        "transparent": true 
    }
    "#;

    let terrain: RawTerrain = serde_json::from_str(text).unwrap();

    assert_eq!(terrain.format_version, None);
    assert_eq!(terrain.description, None);
    assert_eq!(terrain.glyph, None);
}

#[test]
fn a_missing_required_field_is_an_error() {
    let text = r#"{ "id": "base:wall", "walkable": false }"#;

    let result: Result<RawTerrain, _> = serde_json::from_str(text);

    let error = result.unwrap_err();
    assert!(error.to_string().contains("missing field `transparent`"));
}

#[test]
fn a_field_of_the_wrong_type_is_an_error() {
    let text = r#"{ "id": "base:wall", "walkable": "no", "transparent": false }"#;

    let result: Result<RawTerrain, _> = serde_json::from_str(text);

    let error = result.unwrap_err();
    assert!(error.to_string().contains("invalid type"));
}

#[test]
fn an_unknown_field_is_an_error() {
    let text = r#"{ "id": "base:wall", "glpyh": "x", "walkable": false, "transparent": false }"#;

    let result: Result<RawTerrain, _> = serde_json::from_str(text);

    let error = result.unwrap_err();
    assert!(error.to_string().contains("unknown field `glpyh`"));
}

#[test]
fn a_terrain_is_recognised_by_its_category() {
    let text = r##"{ "format_version": 1, "id": "base:wall", "category": "terrain", "description": "Solid.", "glyph": "#", "walkable": false, "transparent": false }"##;

    let object: RawObject = serde_json::from_str(text).unwrap();

    match object {
        RawObject::Terrain(terrain) => assert_eq!(terrain.id, "base:wall"),
    }
}

#[test]
fn an_unknown_category_is_an_error() {
    let text = r#"{ "id": "base:rat", "category": "entity", "glyph": "r" }"#;

    let result: Result<RawObject, _> = serde_json::from_str(text);

    let error = result.unwrap_err();
    assert!(error.to_string().contains("unknown variant `entity`"));
}

#[test]
fn a_missing_category_is_an_error() {
    let text = r#"{ "id": "base:wall", "walkable": false, "transparent": false }"#;

    let result: Result<RawObject, _> = serde_json::from_str(text);

    let error = result.unwrap_err();
    assert!(error.to_string().contains("missing field `category`"));
}

#[test]
fn a_typo_inside_a_categorised_object_is_still_an_error() {
    let text = r#"{ "id": "base:wall", "category": "terrain", "glyph": "x", "glpyh": "x", "walkable": false, "transparent": false }"#;

    let result: Result<RawObject, _> = serde_json::from_str(text);

    let error = result.unwrap_err();
    assert!(error.to_string().contains("unknown field `glpyh`"));
}

#[test]
fn reads_a_pack_file() {
    let text = r#"{ "format_version": 1, "name": "dungeon", "version": "0.1.0",
                    "description": "Rooms.", "depends_on": ["base"] }"#;

    let pack: RawPack = serde_json::from_str(text).unwrap();

    assert_eq!(pack.name, "dungeon");
    assert_eq!(pack.depends_on, vec![String::from("base")]);
}

#[test]
fn depends_on_can_be_left_out() {
    let text = r#"{ "name": "base", "version": "0.1.0" }"#;

    let pack: RawPack = serde_json::from_str(text).unwrap();

    assert!(pack.depends_on.is_empty());
    assert_eq!(pack.description, None);
}

#[test]
fn a_pack_without_a_name_is_an_error() {
    let text = r#"{ "version": "0.1.0" }"#;

    let result: Result<RawPack, _> = serde_json::from_str(text);

    let error = result.unwrap_err();
    assert!(error.to_string().contains("missing field `name`"));
}

#[test]
fn an_unknown_field_in_a_pack_is_an_error() {
    let text = r#"{ "name": "base", "version": "0.1.0", "auther": "me" }"#;

    let result: Result<RawPack, _> = serde_json::from_str(text);

    let error = result.unwrap_err();
    assert!(error.to_string().contains("unknown field `auther`"));
}

#[test]
fn a_terrain_with_an_unknown_field_is_rejected() {
    let text = r##"{"category":"terrain","id":"base:wall","walkable":false,"transparent":false,"colour":"red"}"##;

    let error = serde_json::from_str::<RawObject>(text).unwrap_err();

    assert!(error.to_string().contains("colour"));
}


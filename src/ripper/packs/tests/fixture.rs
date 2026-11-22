//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::path::PathBuf;

use super::super::raw::{RawPack, RawTerrain};
use super::super::read::{LoadedObject, LoadedPack};

//-----------------------------------------------------
// JSON files
//-----------------------------------------------------

pub const WALL: &str = r##"{"category":"terrain","id":"base:wall","glyph":"#","fg_color":"#c8c8c8","bg_color":"#303030","walkable":false,"transparent":false}"##;
pub const EMPTY: &str = r##"{"category":"terrain","id":"base:empty","glyph":".","fg_color":"#404040","bg_color":"#101010","walkable":true,"transparent":true}"##;
pub const SNOW: &str = r##"{"category":"terrain","id":"ice:snow","glyph":"*","fg_color":"#ffffff","bg_color":"#a0b8d0","walkable":true,"transparent":true}"##;

pub const BASE: &str = r#"{"name":"base","version":"0.1.0"}"#;
pub const ICE: &str = r#"{"name":"ice","version":"0.1.0","depends_on":["base"]}"#;

pub const PACK : &str = r#"{"name":"base","version":"0.1.0"}"#;

//-----------------------------------------------------
// Packs
//-----------------------------------------------------

pub fn pack(name: &str) -> LoadedPack {
    LoadedPack {
        dir: PathBuf::from("packs").join(name),
        raw: RawPack {
            format_version: None,
            name: String::from(name),
            version: String::from("0.1.0"),
            description: None,
            depends_on: Vec::new(),
        },
        objects: Vec::new(),
    }
}

pub fn pack_with(name: &str, objects: Vec<LoadedObject>) -> LoadedPack {
    let mut pack = pack(name);
    pack.objects = objects;
    pack
}

//-----------------------------------------------------
// Objects
//-----------------------------------------------------

pub fn object(id: &str, glyph: Option<&str>, file: &str) -> LoadedObject {
    LoadedObject {
        path: PathBuf::from(file),
        terrain: RawTerrain {
            format_version: None,
            id: String::from(id),
            description: None,
            glyph: glyph.map(String::from),
            fg_color: String::from("#ffffff"),
            bg_color: String::from("#000000"),
            walkable: true,
            transparent: true,
        },
    }
}

pub fn plain_object(id: &str, glyph: Option<&str>) -> LoadedObject {
    object(id, glyph, "any.json")
}
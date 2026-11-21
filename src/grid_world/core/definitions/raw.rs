//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use serde::Deserialize;

//-----------------------------------------------------
// Raw Pack
//-----------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawPack {
    pub format_version: Option<u32>,
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

//-----------------------------------------------------
// Raw Terrain
//-----------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawTerrain {
    pub format_version: Option<u32>,
    pub id: String,
    pub description: Option<String>,
    pub glyph: Option<String>,
    pub walkable: bool,
    pub transparent: bool,
}

//-----------------------------------------------------
// Raw Object
//-----------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "category", rename_all = "lowercase")]
pub enum RawObject {
    Terrain(RawTerrain),
}


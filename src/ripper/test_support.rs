//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use crate::ripper::content::store::{Content, TerrainDef, TerrainId};
use crate::ripper::core::color::Color;

//-----------------------------------------------------
// Terrains
//-----------------------------------------------------

pub fn empty() -> TerrainDef {
    TerrainDef {
        id: String::from("base:empty"),
        glyph: Some('.'),
        fg_color: Color::new(64, 64, 64),
        bg_color: Color::new(16, 16, 16),
        walkable: true,
        transparent: true,
    }
}

pub fn wall() -> TerrainDef {
    TerrainDef {
        id: String::from("base:wall"),
        glyph: Some('#'),
        fg_color: Color::new(200, 200, 200),
        bg_color: Color::new(48, 48, 48),
        walkable: false,
        transparent: false,
    }
}

pub fn void() -> TerrainDef {
    TerrainDef {
        id: String::from("base:void"),
        glyph: None,
        fg_color: Color::new(0, 0, 0),
        bg_color: Color::new(0, 0, 0),
        walkable: false,
        transparent: false,
    }
}

pub fn block() -> TerrainDef {
    TerrainDef {
        id: String::from("test:block"),
        glyph: Some('▓'),
        fg_color: Color::new(255, 255, 255),
        bg_color: Color::new(0, 0, 0),
        walkable: false,
        transparent: false,
    }
}

//-----------------------------------------------------
// Content
//-----------------------------------------------------

pub const EMPTY_ID: TerrainId = TerrainId(0);
pub const WALL_ID: TerrainId = TerrainId(1);

pub fn base_content() -> Content {
    Content::new(vec![empty(), wall()])
}
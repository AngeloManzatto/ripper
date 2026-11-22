//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::collections::HashMap;

use super::table::{Named, Table};

//-----------------------------------------------------
// Terrarain ID
//-----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TerrainId(pub u16);

//-----------------------------------------------------
// Terrain Def
//-----------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerrainDef {
    pub id: String,
    pub glyph: Option<char>,
    pub walkable: bool,
    pub transparent: bool,
}

impl Named for TerrainDef {
    fn name(&self) -> &str {
        &self.id
    }
}

//-----------------------------------------------------
// Content
//-----------------------------------------------------

#[derive(Debug)]
pub struct Content {
    terrains: Table<TerrainDef>,
    by_glyph: HashMap<char, TerrainId>,
}

impl Content {
    pub fn new(terrains: Vec<TerrainDef>) -> Content {
        let mut by_glyph = HashMap::new();
        for (position, terrain) in terrains.iter().enumerate() {
            if let Some(glyph) = terrain.glyph {
                by_glyph.insert(glyph, TerrainId(position as u16));
            }
        }
        Content {
            terrains: Table::new(terrains),
            by_glyph,
        }
    }

    pub fn terrains(&self) -> &Table<TerrainDef> {
        &self.terrains
    }

    pub fn terrain(&self, id: TerrainId) -> &TerrainDef {
        self.terrains
            .get(id.0 as usize)
            .expect("terrain id out of range")
    }

    pub fn terrain_id(&self, name: &str) -> Option<TerrainId> {
        self.terrains
            .position_of(name)
            .map(|position| TerrainId(position as u16))
    }

    pub fn terrain_id_by_glyph(&self, glyph: char) -> Option<TerrainId> {
        self.by_glyph.get(&glyph).copied()
    }
}
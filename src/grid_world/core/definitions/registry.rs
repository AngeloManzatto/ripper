//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::collections::HashMap;

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

//-----------------------------------------------------
// Registry
//-----------------------------------------------------

#[derive(Debug)]
pub struct Registry {
    terrains: Vec<TerrainDef>,
    by_name: HashMap<String, TerrainId>,
    by_glyph: HashMap<char, TerrainId>,
}

impl Registry {

    // Create new Registry
    pub fn new(terrains: Vec<TerrainDef>) -> Registry {

        let mut by_name = HashMap::new();
        let mut by_glyph = HashMap::new();

        for (position, terrain) in terrains.iter().enumerate() 
        {

            let id = TerrainId(position as u16);

            by_name.insert(terrain.id.clone(), id);

            if let Some(glyph) = terrain.glyph {
                by_glyph.insert(glyph, id);
            }
        }
        Registry { terrains, by_name, by_glyph }
    }

    // Get terrain id
    pub fn terrain_id(&self, name: &str) -> Option<TerrainId> {
        self.by_name.get(name).copied()
    }

    // Get terrain
    pub fn terrain(&self, id: TerrainId) -> &TerrainDef {
        &self.terrains[id.0 as usize]
    }

    // Get terrain by glyph
    pub fn terrain_id_by_glyph(&self, glyph: char) -> Option<TerrainId> {
        self.by_glyph.get(&glyph).copied()
    }

    // Get glyph by terrain id
    pub fn glyph(&self, id: TerrainId) -> Option<char> {
        self.terrain(id).glyph
    }

    // Count number of terrain definitions
    pub fn terrain_count(&self) -> usize {
        self.terrains.len()
    }

    pub fn walkable(&self, id: TerrainId) -> bool {
        self.terrain(id).walkable
    }

    pub fn transparent(&self, id: TerrainId) -> bool {
        self.terrain(id).transparent
    }

}
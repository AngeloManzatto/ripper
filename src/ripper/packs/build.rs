//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::read::{LoadedObject, LoadedPack};

use crate::ripper::content::store::TerrainDef;

//-----------------------------------------------------
// Build terrais
//-----------------------------------------------------

pub fn build_terrains(packs: &[LoadedPack], order: &[usize]) -> Vec<TerrainDef> {
    let mut terrains = Vec::new();

    for index in order {
        let mut objects: Vec<&LoadedObject> = packs[*index].objects.iter().collect();
        objects.sort_by(|a, b| a.terrain.id.cmp(&b.terrain.id));

        for object in objects {
            let raw = &object.terrain;
            let glyph = match &raw.glyph {
                Some(text) => text.chars().next(),
                None => None,
            };
            terrains.push(TerrainDef {
                id: raw.id.clone(),
                glyph,
                walkable: raw.walkable,
                transparent: raw.transparent,
            });
        }
    }

    terrains
}
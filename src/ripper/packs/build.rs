//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::read::{LoadedObject, LoadedPack};

use crate::ripper::core::color::Color;
use crate::ripper::content::store::TerrainDef;

//-----------------------------------------------------
// Helper
//-----------------------------------------------------

fn parse_color(text: &str) -> Color {
    Color::from_hex(text).expect("colors are checked by validate before build runs")
}

//-----------------------------------------------------
// Build terrais
//-----------------------------------------------------

pub fn build_terrains(packs: &[LoadedPack], order: &[usize]) -> Vec<TerrainDef> 
{

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

            let fg_color = parse_color(&raw.fg_color);
            let bg_color = parse_color(&raw.bg_color);

            terrains.push(TerrainDef {
                id: raw.id.clone(),
                glyph,
                fg_color,
                bg_color,
                walkable: raw.walkable,
                transparent: raw.transparent,
            });
        }
    }

    terrains
}
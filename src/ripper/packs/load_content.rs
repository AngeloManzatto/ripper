//-----------------------------------------------------
// libraries
//-----------------------------------------------------

use std::path::Path;
use super::build::build_terrains;
use super::error::PackError;
use super::read::load_all_packs;
use super::validate::{check_terrain_count, validate};

use crate::ripper::content::store::Content;

//-----------------------------------------------------
// Load content
//-----------------------------------------------------

pub fn load_content(root: &Path) -> Result<Content, Vec<PackError>> {
    let packs = load_all_packs(root)?;

    let mut errors = Vec::new();
    let order = validate(&packs, &mut errors);
    if !errors.is_empty() {
        return Err(errors);
    }

    let terrains = build_terrains(&packs, &order);
    check_terrain_count(terrains.len(), root, &mut errors);
    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(Content::new(terrains))
}
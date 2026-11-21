//-----------------------------------------------------
// libraries
//-----------------------------------------------------

use std::path::Path;
use super::build::build_terrains;
use super::error::DefinitionError;
use super::read::load_all_packs;
use super::registry::Registry;
use super::validate::{check_terrain_count, validate};

//-----------------------------------------------------
// Load registry
//-----------------------------------------------------

pub fn load_registry(root: &Path) -> Result<Registry, Vec<DefinitionError>> {
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

    Ok(Registry::new(terrains))
}
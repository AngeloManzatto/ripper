//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::path::PathBuf;

use super::super::error::format_errors;
use super::super::load_content::load_content;

use crate::ripper::content::store::TerrainId;

//-----------------------------------------------------
// Helpers
//-----------------------------------------------------

fn shipped_packs() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/ripper/assets/packs")
}

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn the_shipped_base_pack_loads() {
    let registry = load_content(&shipped_packs()).unwrap_or_else(|errors| {
        panic!("the shipped definitions are broken:\n{}", format_errors(&errors))
    });

    assert_eq!(registry.terrains().len(), 2);

    let empty = registry.terrain_id("base:empty").unwrap();
    let wall = registry.terrain_id("base:wall").unwrap();

    assert_eq!(empty, TerrainId(0));
    assert_eq!(wall, TerrainId(1));
    assert!(registry.terrain(empty).walkable);
    assert!(registry.terrain(empty).transparent);
    assert!(!registry.terrain(wall).walkable);
    assert!(!registry.terrain(wall).transparent);
    assert_eq!(registry.terrain(empty).glyph, Some('.'));
    assert_eq!(registry.terrain(wall).glyph, Some('#'));
}
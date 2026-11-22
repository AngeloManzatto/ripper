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
    let content = load_content(&shipped_packs()).unwrap_or_else(|errors| {
        panic!("the shipped definitions are broken:\n{}", format_errors(&errors))
    });

    assert_eq!(content.terrains().len(), 2);

    let empty = content.terrain_id("base:empty").unwrap();
    let wall = content.terrain_id("base:wall").unwrap();

    assert_eq!(empty, TerrainId(0));
    assert_eq!(wall, TerrainId(1));
    assert!(content.terrain(empty).walkable);
    assert!(content.terrain(empty).transparent);
    assert!(!content.terrain(wall).walkable);
    assert!(!content.terrain(wall).transparent);
    assert_eq!(content.terrain(empty).glyph, Some('.'));
    assert_eq!(content.terrain(wall).glyph, Some('#'));
}
//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[cfg(test)]
mod tests {

    use crate::grid_world::core::definitions::build::build_terrains;
    use crate::grid_world::core::definitions::read::{LoadedObject, LoadedPack};
    use crate::grid_world::core::definitions::raw::{RawPack, RawTerrain};
    use crate::grid_world::core::definitions::registry::TerrainDef;
    use std::path::PathBuf;

    fn pack(name: &str, objects: Vec<LoadedObject>) -> LoadedPack {
        LoadedPack {
            dir: PathBuf::from("packs").join(name),
            raw: RawPack {
                format_version: None,
                name: String::from(name),
                version: String::from("0.1.0"),
                description: None,
                depends_on: Vec::new(),
            },
            objects,
        }
    }

    fn object(id: &str, glyph: Option<&str>) -> LoadedObject {
        LoadedObject {
            path: PathBuf::from("any.json"),
            terrain: RawTerrain {
                format_version: None,
                id: String::from(id),
                description: None,
                glyph: glyph.map(String::from),
                walkable: true,
                transparent: true,
            },
        }
    }

    fn ids(terrains: &[TerrainDef]) -> Vec<String> {
        let mut ids = Vec::new();
        for terrain in terrains {
            ids.push(terrain.id.clone());
        }
        ids
    }

    #[test]
    fn terrains_follow_the_load_order_then_the_id() {
        let packs = vec![
            pack("ice", vec![object("ice:snow", Some("*"))]),
            pack("base", vec![object("base:wall", Some("#")), object("base:empty", Some("."))]),
        ];

        let terrains = build_terrains(&packs, &[1, 0]);

        assert_eq!(ids(&terrains), vec!["base:empty", "base:wall", "ice:snow"]);
    }

    #[test]
    fn the_glyph_becomes_a_character_and_a_missing_glyph_stays_missing() {
        let packs = vec![pack("base", vec![object("base:empty", None), object("base:wall", Some("#"))])];

        let terrains = build_terrains(&packs, &[0]);

        assert_eq!(terrains[0].glyph, None);
        assert_eq!(terrains[1].glyph, Some('#'));
    }

    #[test]
    fn the_flags_are_copied() {
        let mut solid = object("base:wall", Some("#"));
        solid.terrain.walkable = false;
        solid.terrain.transparent = false;
        let packs = vec![pack("base", vec![solid])];

        let terrains = build_terrains(&packs, &[0]);

        assert!(!terrains[0].walkable);
        assert!(!terrains[0].transparent);
    }

    #[test]
    fn no_packs_give_no_terrains() {
        assert!(build_terrains(&[], &[]).is_empty());
    }

    #[test]
    fn the_order_of_the_files_does_not_change_the_result() {
        let one = vec![pack("base", vec![object("base:a", None), object("base:b", None)])];
        let other = vec![pack("base", vec![object("base:b", None), object("base:a", None)])];

        assert_eq!(build_terrains(&one, &[0]), build_terrains(&other, &[0]));
    }

}
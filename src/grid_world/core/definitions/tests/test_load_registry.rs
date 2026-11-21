//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[cfg(test)]
mod tests {

    use std::fs;
    use std::path::PathBuf;

    use crate::grid_world::core::definitions::error::{DefinitionError, format_errors};
    use crate::grid_world::core::definitions::registry::TerrainId;
    use crate::grid_world::core::definitions::validate::{check_terrain_count, MAX_TERRAINS};

    use crate::grid_world::core::definitions::load_registry::load_registry;

    fn fresh_temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_pack(root: &PathBuf, folder: &str, pack_json: &str, objects: &[(&str, &str)]) {
        let dir = root.join(folder);
        fs::create_dir_all(dir.join("terrain")).unwrap();
        fs::write(dir.join("pack.json"), pack_json).unwrap();
        for (file, content) in objects {
            fs::write(dir.join("terrain").join(file), content).unwrap();
        }
    }

    const WALL: &str = r##"{"category":"terrain","id":"base:wall","glyph":"#","walkable":false,"transparent":false}"##;
    const EMPTY: &str = r##"{"category":"terrain","id":"base:empty","glyph":".","walkable":true,"transparent":true}"##;
    const BASE: &str = r#"{"name":"base","version":"0.1.0"}"#;

    #[test]
    fn a_good_folder_gives_a_registry_with_numbered_terrains() {
        let root = fresh_temp_dir("ripper_registry_ok");
        write_pack(&root, "base", BASE, &[("wall.json", WALL), ("empty.json", EMPTY)]);

        let registry = load_registry(&root).unwrap();

        assert_eq!(registry.terrain_count(), 2);
        assert_eq!(registry.terrain_id("base:empty"), Some(TerrainId(0)));
        assert_eq!(registry.terrain_id("base:wall"), Some(TerrainId(1)));
        assert!(registry.walkable(TerrainId(0)));
        assert!(!registry.walkable(TerrainId(1)));
    }

    #[test]
    fn a_pack_that_depends_on_another_is_numbered_after_it() {
        let root = fresh_temp_dir("ripper_registry_two_packs");
        let ice = r#"{"name":"ice","version":"0.1.0","depends_on":["base"]}"#;
        let snow = r##"{"category":"terrain","id":"ice:snow","glyph":"*","walkable":true,"transparent":true}"##;
        write_pack(&root, "ice", ice, &[("snow.json", snow)]);
        write_pack(&root, "base", BASE, &[("wall.json", WALL)]);

        let registry = load_registry(&root).unwrap();

        assert_eq!(registry.terrain_id("base:wall"), Some(TerrainId(0)));
        assert_eq!(registry.terrain_id("ice:snow"), Some(TerrainId(1)));
    }

    #[test]
    fn a_file_that_cannot_be_read_stops_the_load() {
        let root = fresh_temp_dir("ripper_registry_unreadable");
        write_pack(&root, "base", BASE, &[("wall.json", WALL), ("empty.json", "not json")]);

        assert!(load_registry(&root).is_err());
    }

    #[test]
    fn a_broken_rule_stops_the_load_and_every_rule_problem_is_reported() {
        let root = fresh_temp_dir("ripper_registry_rules");
        write_pack(&root, "base", BASE, &[("wall.json", WALL), ("wall_copy.json", WALL)]);

        let errors = load_registry(&root).unwrap_err();

        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn a_missing_root_is_an_error() {
        let root = std::env::temp_dir().join("ripper_registry_missing");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(load_registry(&root).unwrap_err().len(), 1);
    }

    #[test]
    fn the_limit_of_terrains_is_enforced() {
        let root = PathBuf::from("packs");
        let mut errors: Vec<DefinitionError> = Vec::new();

        check_terrain_count(MAX_TERRAINS, &root, &mut errors);
        assert!(errors.is_empty());

        check_terrain_count(MAX_TERRAINS + 1, &root, &mut errors);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].path, root);
        assert!(errors[0].message.contains("too many terrains"));
    }

    fn shipped_packs() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/grid_world/assets/packs")
    }

    #[test]
    fn the_shipped_base_pack_loads() {
        let registry = load_registry(&shipped_packs()).unwrap_or_else(|errors| {
            panic!("the shipped definitions are broken:\n{}", format_errors(&errors))
        });

        assert_eq!(registry.terrain_count(), 2);

        let empty = registry.terrain_id("base:empty").unwrap();
        let wall = registry.terrain_id("base:wall").unwrap();

        assert_eq!(empty, TerrainId(0));
        assert_eq!(wall, TerrainId(1));
        assert!(registry.walkable(empty));
        assert!(registry.transparent(empty));
        assert!(!registry.walkable(wall));
        assert!(!registry.transparent(wall));
        assert_eq!(registry.terrain(empty).glyph, Some('.'));
        assert_eq!(registry.terrain(wall).glyph, Some('#'));
    }

}
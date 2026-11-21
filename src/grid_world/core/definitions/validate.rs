//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::path::Path;
use std::collections::{HashMap, HashSet};

use super::error::DefinitionError;
use super::read::{LoadedObject, LoadedPack};
use super::SUPPORTED_FORMAT_VERSION;

//-----------------------------------------------------
// Constants
//-----------------------------------------------------

pub const MAX_TERRAINS: usize = u16::MAX as usize;

//-----------------------------------------------------
// Check ID
//-----------------------------------------------------

pub fn split_id(id: &str) -> Option<(&str, &str)> {
    match id.split_once(':') {
        Some((prefix, name)) if !prefix.is_empty() && !name.is_empty() && !name.contains(':') => {
            Some((prefix, name))
        }
        _ => None,
    }
}

pub fn check_id(id: &str, pack_name: &str, path: &Path, errors: &mut Vec<DefinitionError>) {
    let prefix = match split_id(id) {
        Some((prefix, _name)) => prefix,
        None => {
            errors.push(DefinitionError::new(
                path.to_path_buf(),
                format!("the id `{id}` must look like `<pack>:<name>`"),
            ));
            return;
        }
    };

    if prefix != pack_name {
        errors.push(DefinitionError::new(
            path.to_path_buf(),
            format!("the id `{id}` must start with the pack name `{pack_name}:`"),
        ));
    }
}

pub fn check_unique_ids(packs: &[LoadedPack], errors: &mut Vec<DefinitionError>) {
    let mut seen: HashMap<&str, &Path> = HashMap::new();

    for pack in packs {
        for object in &pack.objects {
            let id = object.terrain.id.as_str();

            match seen.get(id) {
                Some(first) => errors.push(DefinitionError::new(
                    object.path.clone(),
                    format!("the id `{id}` is already defined in {}", first.display()),
                )),
                None => {
                    seen.insert(id, object.path.as_path());
                }
            }
        }
    }
}

//-----------------------------------------------------
// Check glyph
//-----------------------------------------------------

pub fn check_glyph(glyph: &Option<String>, path: &Path, errors: &mut Vec<DefinitionError>) {

    if let Some(glyph) = glyph {

        let mut chars = glyph.chars();

        let first_char = chars.next();
        let second_chat = chars.next();

        // Validate for a single char only
        let valid = match (first_char, second_chat) 
        {
            (Some(c), None) => !c.is_whitespace() && !c.is_control(),
            _ => false,
        };
        
        if !valid {
            errors.push(DefinitionError::new(
                path.to_path_buf(),
                format!("the glyph {glyph:?} must be exactly one visible character"),
            ));
        }
    }
}

pub fn check_unique_glyphs(packs: &[LoadedPack], errors: &mut Vec<DefinitionError>) {
    let mut seen: HashMap<char, &Path> = HashMap::new();

    for pack in packs {
        for object in &pack.objects {
            let Some(text) = &object.terrain.glyph else {
                continue;
            };
            let Some(glyph) = text.chars().next() else {
                continue;
            };

            match seen.get(&glyph) {
                Some(first) => errors.push(DefinitionError::new(
                    object.path.clone(),
                    format!("the glyph `{glyph}` is already used in {}", first.display()),
                )),
                None => {
                    seen.insert(glyph, object.path.as_path());
                }
            }
        }
    }
}

//-----------------------------------------------------
// Check name characters
//-----------------------------------------------------

pub fn check_name_characters(name: &str, path: &Path, errors: &mut Vec<DefinitionError>) {
    let valid = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');

    if !valid {
        errors.push(DefinitionError::new(
            path.to_path_buf(),
            format!("the name `{name}` may only use a-z, 0-9 and _"),
        ));
    }
}

//-----------------------------------------------------
// Check name matches file
//-----------------------------------------------------

pub fn check_name_matches_file(name: &str, path: &Path, errors: &mut Vec<DefinitionError>) {
    let matches = match path.file_stem() {
        Some(stem) => stem == name,
        None => false,
    };

    if !matches {
        errors.push(DefinitionError::new(
            path.to_path_buf(),
            format!("the name `{name}` must be the file name: expected `{name}.json`"),
        ));
    }
}

//-----------------------------------------------------
// Check terrain count
//-----------------------------------------------------

pub fn check_terrain_count(count: usize, root: &Path, errors: &mut Vec<DefinitionError>) {
    if count > MAX_TERRAINS {
        errors.push(DefinitionError::new(
            root.to_path_buf(),
            format!("too many terrains: {count} found, the limit is {MAX_TERRAINS}"),
        ));
    }
}

//-----------------------------------------------------
// Validate object
//-----------------------------------------------------

pub fn validate_object(
    pack: &LoadedPack,
    object: &LoadedObject,
    errors: &mut Vec<DefinitionError>,
) {
    let terrain = &object.terrain;

    check_format_version(terrain.format_version, &object.path, errors);
    check_glyph(&terrain.glyph, &object.path, errors);
    check_id(&terrain.id, &pack.raw.name, &object.path, errors);

    if let Some((_prefix, name)) = split_id(&terrain.id) {
        check_name_characters(name, &object.path, errors);
        check_name_matches_file(name, &object.path, errors);
    }
}

//-----------------------------------------------------
// Check format version
//-----------------------------------------------------

pub fn check_format_version(version: Option<u32>, path: &Path, errors: &mut Vec<DefinitionError>) {
    if let Some(version) = version {
        if version != SUPPORTED_FORMAT_VERSION {
            errors.push(DefinitionError::new(
                path.to_path_buf(),
                format!(
                    "format_version {version} is not supported (this engine reads version {SUPPORTED_FORMAT_VERSION})"
                ),
            ));
        }
    }
}

//-----------------------------------------------------
// Check dependencies exist
//-----------------------------------------------------

pub fn check_dependencies_exist(packs: &[LoadedPack], errors: &mut Vec<DefinitionError>) {
    let mut names: HashSet<&str> = HashSet::new();
    for pack in packs {
        names.insert(pack.raw.name.as_str());
    }

    for pack in packs {
        for dependency in &pack.raw.depends_on {
            if !names.contains(dependency.as_str()) {
                errors.push(DefinitionError::new(
                    pack.dir.join("pack.json"),
                    format!(
                        "the pack `{}` depends on `{}`, but there is no such pack",
                        pack.raw.name, dependency
                    ),
                ));
            }
        }
    }
}

//-----------------------------------------------------
// Validate pack
//-----------------------------------------------------

pub fn validate_pack(pack: &LoadedPack, errors: &mut Vec<DefinitionError>) {
    let pack_path = pack.dir.join("pack.json");

    let folder_matches = match pack.dir.file_name() {
        Some(folder) => folder == pack.raw.name.as_str(),
        None => false,
    };

    if !folder_matches {
        errors.push(DefinitionError::new(
            pack_path.clone(),
            format!(
                "the pack name `{0}` must be the folder name: expected a folder called `{0}`",
                pack.raw.name
            ),
        ));
    }

    check_format_version(pack.raw.format_version, &pack_path, errors);
}

//-----------------------------------------------------
// Load order
//-----------------------------------------------------

pub fn load_order(packs: &[LoadedPack], errors: &mut Vec<DefinitionError>) -> Vec<usize> {
    let mut names: HashSet<&str> = HashSet::new();
    for pack in packs {
        names.insert(pack.raw.name.as_str());
    }

    let mut order: Vec<usize> = Vec::new();
    let mut placed: HashSet<&str> = HashSet::new();

    loop {
        let mut next: Option<usize> = None;

        for (index, pack) in packs.iter().enumerate() {
            if placed.contains(pack.raw.name.as_str()) {
                continue;
            }

            let mut ready = true;
            for dependency in &pack.raw.depends_on {
                let known = names.contains(dependency.as_str());
                if known && !placed.contains(dependency.as_str()) {
                    ready = false;
                }
            }
            if !ready {
                continue;
            }

            let better = match next {
                Some(best) => pack.raw.name < packs[best].raw.name,
                None => true,
            };
            if better {
                next = Some(index);
            }
        }

        match next {
            Some(index) => {
                placed.insert(packs[index].raw.name.as_str());
                order.push(index);
            }
            None => break,
        }
    }

    for pack in packs {
        if !placed.contains(pack.raw.name.as_str()) {
            errors.push(DefinitionError::new(
                pack.dir.join("pack.json"),
                format!(
                    "the pack `{}` cannot be ordered: it is in a dependency cycle or depends on a pack that is",
                    pack.raw.name
                ),
            ));
        }
    }

    order
}

//-----------------------------------------------------
// Validate
//-----------------------------------------------------

pub fn validate(packs: &[LoadedPack], errors: &mut Vec<DefinitionError>) -> Vec<usize> {
    for pack in packs {
        validate_pack(pack, errors);
        for object in &pack.objects {
            validate_object(pack, object, errors);
        }
    }

    check_unique_ids(packs, errors);
    check_dependencies_exist(packs, errors);
    check_unique_glyphs(packs, errors);
    load_order(packs, errors)
}
//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use super::error::DefinitionError;
use super::raw::{RawObject, RawPack, RawTerrain};

use serde::de::DeserializeOwned;

//-----------------------------------------------------
// Loaded Object
//-----------------------------------------------------

#[derive(Debug)]
pub struct LoadedObject {
    pub path: PathBuf,
    pub terrain: RawTerrain,
}

//-----------------------------------------------------
// Loaded Pack
//-----------------------------------------------------

#[derive(Debug)]
pub struct LoadedPack {
    pub dir: PathBuf,
    pub raw: RawPack,
    pub objects: Vec<LoadedObject>,
}

//-----------------------------------------------------
// Sorted entries
//-----------------------------------------------------

pub fn sorted_entries(dir: &Path) -> Result<Vec<PathBuf>, DefinitionError> {

    // Directory folder
    let read = match fs::read_dir(dir) {
        Ok(read) => read,
        Err(error) => {
            return Err(DefinitionError::new(
                dir.to_path_buf(),
                format!("cannot read the folder: {error}"),
            ));
        }
    };

    // Load paths from directory folder
    let mut paths = Vec::new();
    for entry in read {
        match entry {
            Ok(entry) => paths.push(entry.path()),
            Err(error) => {
                return Err(DefinitionError::new(
                    dir.to_path_buf(),
                    format!("cannot read a folder entry: {error}"),
                ));
            }
        }
    }

    // Sort paths 
    paths.sort();
    Ok(paths)
}

//-----------------------------------------------------
// Read Json
//-----------------------------------------------------

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, DefinitionError> {

    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            return Err(DefinitionError::new(
                path.to_path_buf(),
                format!("cannot read the file: {error}"),
            ));
        }
    };

    match serde_json::from_str::<T>(&text) {
        Ok(value) => Ok(value),
        Err(error) => Err(DefinitionError::new(
            path.to_path_buf(),
            format!("invalid definition: {error}"),
        )),
    }
}

//-----------------------------------------------------
// Read Object
//-----------------------------------------------------

pub fn read_object(path: &Path) -> Result<RawObject, DefinitionError> {
    read_json(path)
}

//-----------------------------------------------------
// Read Pack
//-----------------------------------------------------

pub fn read_pack(path: &Path) -> Result<RawPack, DefinitionError> {
    read_json(path)
}

//-----------------------------------------------------
// Collect json files
//-----------------------------------------------------

pub fn collect_json_files(
    dir: &Path,
    found: &mut Vec<PathBuf>,
    errors: &mut Vec<DefinitionError>,
) {
    let entries = match sorted_entries(dir) {
        Ok(entries) => entries,
        Err(error) => {
            errors.push(error);
            return;
        }
    };

    for path in entries {
        if path.is_dir() {
            collect_json_files(&path, found, errors);
        } else {
            let is_json = match path.extension() {
                Some(extension) => extension == "json",
                None => false,
            };
            if is_json {
                found.push(path);
            }
        }
    }
}

//-----------------------------------------------------
// Find json files
//-----------------------------------------------------

pub fn find_json_files(dir: &Path) -> Result<Vec<PathBuf>, Vec<DefinitionError>> {
    let mut found = Vec::new();
    let mut errors = Vec::new();

    collect_json_files(dir, &mut found, &mut errors);

    if errors.is_empty() {
        Ok(found)
    } else {
        Err(errors)
    }
}

//-----------------------------------------------------
// Load pack
//-----------------------------------------------------

pub fn load_pack(dir: &Path) -> Result<LoadedPack, Vec<DefinitionError>> {

    // Pack json file
    let pack_path = dir.join("pack.json");

    // Check if file exists
    if !pack_path.is_file() {
        return Err(vec![DefinitionError::new(
            dir.to_path_buf(),
            String::from("the pack folder has no pack.json"),
        )]);
    }

    let mut errors = Vec::new();

    // Load raw pack json file
    let raw = match read_pack(&pack_path) {
        Ok(raw) => Some(raw),
        Err(error) => {
            errors.push(error);
            None
        }
    };

    // Get all files inside base directory
    let files = match find_json_files(dir) {
        Ok(files) => files,
        Err(found_errors) => {
            errors.extend(found_errors);
            Vec::new()
        }
    };

    let mut objects = Vec::new();

    for path in files {

        // Ignore base pack.json
        if path == pack_path {
            continue;
        }

        match read_object(&path) {
            Ok(RawObject::Terrain(terrain)) => objects.push(LoadedObject { path, terrain }),
            Err(error) => errors.push(error),
        }
    }

    let raw = match raw {
        Some(raw) => raw,
        None => return Err(errors),
    };
    
    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(LoadedPack { dir: dir.to_path_buf(), raw, objects })
}

//-----------------------------------------------------
// Load all packs
//-----------------------------------------------------

pub fn load_all_packs(root: &Path) -> Result<Vec<LoadedPack>, Vec<DefinitionError>> {
    let entries = match sorted_entries(root) {
        Ok(entries) => entries,
        Err(error) => return Err(vec![error]),
    };

    let mut packs = Vec::new();
    let mut errors = Vec::new();

    for path in entries {
        if !path.is_dir() {
            continue;
        }
        match load_pack(&path) {
            Ok(pack) => packs.push(pack),
            Err(pack_errors) => errors.extend(pack_errors),
        }
    }

    if errors.is_empty() {
        Ok(packs)
    } else {
        Err(errors)
    }
}
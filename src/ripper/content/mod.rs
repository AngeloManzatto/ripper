//! Content: what the engine knows about the game's stuff, in memory.
//!
//! Words used across the engine:
//! - Pack: a folder on disk with pack.json and one subfolder per kind of content.
//! - Raw: the shape of a file before any checks (see packs/raw.rs).
//! - Def: a checked, in-memory definition (TerrainDef).
//! - Id: the number of a def inside its table (TerrainId).
//! - Table: all defs of one kind, in order, with lookups by name and position.
//! - Content: the bundle of all tables, plus the lookups that need typed ids.
//!
//! Knows nothing about files or JSON. May use: core. Used by: packs, world, rules, render.
//!
//! 
//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

pub mod store;
pub mod table;

#[cfg(test)]
pub mod tests;
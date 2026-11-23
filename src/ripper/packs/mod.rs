//! Packs: loading game content from disk.
//!
//! Entry point: `load_content` (see load_content.rs).
//! Steps, in order: read.rs (files → raw.rs shapes), validate.rs (rules),
//! build.rs (raw → definitions), then Content::new.
//! May use: core, content. Must never be used by: world, rules, render.
//! Errors are collected, never silently skipped (error.rs).

//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

pub mod error;
pub mod raw;
pub mod read;
pub mod validate;
pub mod build;
mod load_content;

#[cfg(test)]
pub mod tests;

//-----------------------------------------------------
// Constants
//-----------------------------------------------------

pub const SUPPORTED_FORMAT_VERSION: u32 = 1;

pub use load_content::load_content;
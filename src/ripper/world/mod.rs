//! World: a level in memory, built from text and the registry.
//!
//! May use: core, content. Must never use: packs (only startup code does).

//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

pub mod layout;
pub mod level;
pub mod tests;
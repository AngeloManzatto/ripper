//! World: a level in memory, built from text and the content.
//!
//! May use: core, content. Must never use: packs (only startup code does).

//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

pub mod layout;
pub mod level;

#[cfg(test)]
pub mod tests;
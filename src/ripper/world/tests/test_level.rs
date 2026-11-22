//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::sync::Arc;

use super::super::layout::LayoutError;
use super::super::level::Level;

use crate::ripper::content::store::{Content, TerrainDef, TerrainId};
use crate::ripper::render::terminal::draw_terrain;

//-----------------------------------------------------
// Helpers
//-----------------------------------------------------

fn empty() -> TerrainDef {
    TerrainDef { id: String::from("base:empty"), glyph: Some('.'), walkable: true, transparent: true }
}

fn wall() -> TerrainDef {
    TerrainDef { id: String::from("base:wall"), glyph: Some('#'), walkable: false, transparent: false }
}

fn content() -> Arc<Content> {
    Arc::new(Content::new(vec![empty(), wall()]))
}

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn a_level_is_built_from_a_layout() {
    let level = Level::from_layout("#.\n.#", content()).unwrap();

    assert_eq!(level.terrain().width(), 2);
    assert_eq!(level.terrain().get((0, 0)), Some(&TerrainId(1)));
}

#[test]
fn a_bad_layout_gives_the_layout_errors() {
    let errors = Level::from_layout("#?", content()).unwrap_err();

    assert_eq!(errors, vec![LayoutError::UnknownGlyph { row: 0, col: 1, glyph: '?' }]);
}

#[test]
fn a_level_answers_questions_through_its_content() {
    let level = Level::from_layout("#.", content()).unwrap();

    let id = *level.terrain().get((0, 0)).unwrap();

    assert!(!level.content().terrain(id).walkable);
}

#[test]
fn many_levels_share_one_content_and_do_not_copy_it() {
    let shared = content();

    let first = Level::from_layout("#", Arc::clone(&shared)).unwrap();
    let second = Level::from_layout(".", Arc::clone(&shared)).unwrap();

    assert_eq!(Arc::strong_count(&shared), 3);
    drop(first);
    assert_eq!(Arc::strong_count(&shared), 2);
    drop(second);
    assert_eq!(Arc::strong_count(&shared), 1);
}

#[test]
fn a_level_can_be_drawn_from_its_own_parts() {
    let level = Level::from_layout("#.\n.#", content()).unwrap();

    assert_eq!(draw_terrain(level.terrain(), level.content()), "#.\n.#");
}
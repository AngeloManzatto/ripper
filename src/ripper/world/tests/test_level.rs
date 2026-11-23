//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::sync::Arc;

use super::super::layout::LayoutError;
use super::super::level::Level;

use crate::ripper::render::paint::paint_level;
use crate::ripper::render::terminal::frame_to_text;

use crate::ripper::test_support::{
    base_content, 
    WALL_ID
};

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn a_level_is_built_from_a_layout() {

    let content = Arc::new(base_content());

    let level = Level::from_layout("#.\n.#", content).unwrap();

    assert_eq!(level.terrain().width(), 2);
    assert_eq!(level.terrain().get((0, 0)), Some(&WALL_ID));
}

#[test]
fn a_bad_layout_gives_the_layout_errors() {

    let content = Arc::new(base_content());

    let errors = Level::from_layout("#?", content).unwrap_err();

    assert_eq!(errors, vec![LayoutError::UnknownGlyph { row: 0, col: 1, glyph: '?' }]);
}

#[test]
fn a_level_answers_questions_through_its_content() {

    let content = Arc::new(base_content());

    let level = Level::from_layout("#.", content).unwrap();

    let id = *level.terrain().get((0, 0)).unwrap();

    assert!(!level.content().terrain(id).walkable);
}

#[test]
fn many_levels_share_one_content_and_do_not_copy_it() {

    let shared = Arc::new(base_content());

    let first = Level::from_layout("#", Arc::clone(&shared)).unwrap();
    let second = Level::from_layout(".", Arc::clone(&shared)).unwrap();

    assert_eq!(Arc::strong_count(&shared), 3);
    drop(first);
    assert_eq!(Arc::strong_count(&shared), 2);
    drop(second);
    assert_eq!(Arc::strong_count(&shared), 1);
}

#[test]
fn a_level_can_be_painted_from_its_own_parts() {

    let content = Arc::new(base_content());

    let level = Level::from_layout("#.\n.#", content).unwrap();

    assert_eq!(frame_to_text(&paint_level(&level)), "#.\n.#");
}
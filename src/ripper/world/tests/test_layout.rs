//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::super::layout::{parse_layout, LayoutError};

use crate::ripper::content::store::{Content, TerrainDef, TerrainId};

//-----------------------------------------------------
// Helpers
//-----------------------------------------------------

fn empty() -> TerrainDef {
    TerrainDef { id: String::from("base:empty"), glyph: Some('.'), walkable: true, transparent: true }
}

fn wall() -> TerrainDef {
    TerrainDef { id: String::from("base:wall"), glyph: Some('#'), walkable: false, transparent: false }
}

fn block() -> TerrainDef {
    TerrainDef { id: String::from("test:block"), glyph: Some('▓'), walkable: false, transparent: false }
}

fn registry() -> Content {
    Content::new(vec![empty(), wall()])
}

const EMPTY_ID: TerrainId = TerrainId(0);
const WALL_ID: TerrainId = TerrainId(1);

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn a_layout_becomes_a_grid_of_terrain_ids() {
    let grid = parse_layout("#.\n.#", &registry()).unwrap();

    assert_eq!(grid.get((0, 0)), Some(&WALL_ID));
    assert_eq!(grid.get((0, 1)), Some(&EMPTY_ID));
    assert_eq!(grid.get((1, 0)), Some(&EMPTY_ID));
    assert_eq!(grid.get((1, 1)), Some(&WALL_ID));
}

#[test]
fn the_grid_has_the_size_of_the_text() {
    let grid = parse_layout("###\n...", &registry()).unwrap();

    assert_eq!(grid.width(), 3);
    assert_eq!(grid.height(), 2);
}

#[test]
fn rows_come_before_columns() {
    let grid = parse_layout("#..\n...", &registry()).unwrap();

    assert_eq!(grid.get((0, 0)), Some(&WALL_ID));
    assert_eq!(grid.get((1, 2)), Some(&EMPTY_ID));
    assert_eq!(grid.get((2, 0)), None);
}

#[test]
fn a_final_newline_adds_no_row() {
    let grid = parse_layout("##\n..\n", &registry()).unwrap();

    assert_eq!(grid.height(), 2);
}

#[test]
fn windows_line_endings_are_read_like_any_other() {
    let grid = parse_layout("##\r\n..\r\n", &registry()).unwrap();

    assert_eq!(grid.width(), 2);
    assert_eq!(grid.height(), 2);
}

#[test]
fn a_glyph_wider_than_one_byte_is_still_one_column() {
    let registry = Content::new(vec![empty(), wall(), block()]);

    let grid = parse_layout("▓▓\n▓.", &registry).unwrap();

    assert_eq!(grid.width(), 2);
    assert_eq!(grid.get((0, 1)), Some(&TerrainId(2)));
}

#[test]
fn no_text_is_an_error() {
    assert_eq!(parse_layout("", &registry()).unwrap_err(), vec![LayoutError::Empty]);
}

#[test]
fn only_a_blank_line_is_an_error() {
    assert_eq!(parse_layout("\n", &registry()).unwrap_err(), vec![LayoutError::Empty]);
}

#[test]
fn a_row_of_another_length_is_reported_with_its_row_and_sizes() {
    let errors = parse_layout("###\n##", &registry()).unwrap_err();

    assert_eq!(errors, vec![LayoutError::RaggedRow { row: 1, expected: 3, found: 2 }]);
}

#[test]
fn every_row_of_another_length_is_reported() {
    let errors = parse_layout("###\n##\n####", &registry()).unwrap_err();

    assert_eq!(errors.len(), 2);
}

#[test]
fn an_unknown_glyph_is_reported_with_its_position() {
    let errors = parse_layout("#.\n.?", &registry()).unwrap_err();

    assert_eq!(errors, vec![LayoutError::UnknownGlyph { row: 1, col: 1, glyph: '?' }]);
}

#[test]
fn every_unknown_glyph_is_reported_together() {
    let errors = parse_layout("?.\n!#", &registry()).unwrap_err();

    assert_eq!(errors.len(), 2);
}

#[test]
fn a_ragged_row_and_an_unknown_glyph_are_reported_together() {
    let errors = parse_layout("#?\n###", &registry()).unwrap_err();

    assert_eq!(errors.len(), 2);
}
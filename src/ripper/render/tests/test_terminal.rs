//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::super::terminal::draw_terrain;

use crate::ripper::content::store::{Content, TerrainDef, TerrainId};
use crate::ripper::core::grid::Grid;
use crate::ripper::world::layout::parse_layout;

//-----------------------------------------------------
// Helpers
//-----------------------------------------------------

fn empty() -> TerrainDef {
    TerrainDef { id: String::from("base:empty"), glyph: Some('.'), walkable: true, transparent: true }
}

fn wall() -> TerrainDef {
    TerrainDef { id: String::from("base:wall"), glyph: Some('#'), walkable: false, transparent: false }
}

fn void() -> TerrainDef {
    TerrainDef { id: String::from("base:void"), glyph: None, walkable: false, transparent: false }
}

fn block() -> TerrainDef {
    TerrainDef { id: String::from("test:block"), glyph: Some('▓'), walkable: false, transparent: false }
}

fn content() -> Content {
    Content::new(vec![empty(), wall()])
}

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn a_grid_is_drawn_with_the_glyph_of_each_terrain() {
    let mut grid = Grid::new(2, 2, TerrainId(0));
    *grid.get_mut((0, 0)).unwrap() = TerrainId(1);

    assert_eq!(draw_terrain(&grid, &content()), "#.\n..");
}

#[test]
fn a_layout_drawn_again_gives_the_same_text() {
    let content = content();

    for text in ["#", "###\n#.#\n###", ".#.\n#.#", "....\n....\n####"] {
        let grid = parse_layout(text, &content).unwrap();

        assert_eq!(draw_terrain(&grid, &content), text);
    }
}

#[test]
fn the_drawing_has_no_final_newline() {
    let grid = parse_layout("##\n..\n", &content()).unwrap();

    assert_eq!(draw_terrain(&grid, &content()), "##\n..");
}

#[test]
fn a_terrain_without_a_glyph_is_drawn_as_a_space() {
    let content = Content::new(vec![empty(), wall(), void()]);
    let grid = Grid::new(2, 1, TerrainId(2));

    assert_eq!(draw_terrain(&grid, &content), "  ");
}

#[test]
fn a_glyph_wider_than_one_byte_is_drawn_whole() {
    let content = Content::new(vec![empty(), wall(), block()]);
    let grid = parse_layout("▓▓\n▓.", &content).unwrap();

    assert_eq!(draw_terrain(&grid, &content), "▓▓\n▓.");
}

#[test]
fn the_drawing_follows_the_content_not_the_number() {
    let swapped = Content::new(vec![wall(), empty()]);
    let grid = Grid::new(1, 1, TerrainId(0));

    assert_eq!(draw_terrain(&grid, &swapped), "#");
}

#[test]
fn look_at_it() {
    let content = Content::new(vec![empty(), wall()]);
    let layout = "##########\n#........#\n#..####..#\n#..#..#..#\n#........#\n##########";
    let grid = parse_layout(layout, &content).unwrap();

    println!("\n{}\n", draw_terrain(&grid, &content));
}
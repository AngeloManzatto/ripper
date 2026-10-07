//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use crate::grid_world::layout;
use crate::grid_world::pathfinding::{geometry,bfs};

//-----------------------------------------------------
// Is layout reachable
//-----------------------------------------------------

pub fn is_layout_reachable(layout: &str) -> bool {
    let grid = layout::parsers::layout_to_grid(layout);

    let start = match layout::parsers::find_char(&grid, 'P') {
        Some(pos) => pos,
        None => return false,
    };
    let target = match layout::parsers::find_char(&grid, 'G') {
        Some(pos) => pos,
        None => return false,
    };

    bfs::bfs_search(grid, start, target, vec!['#', 'T']).is_some()
}

//-----------------------------------------------------
// Is player far from goal
//-----------------------------------------------------

pub fn is_player_far_from_goal(layout: &str, min_distance: usize) -> bool {

    let grid = layout::parsers::layout_to_grid(layout);

    let start = match layout::parsers::find_char(&grid, 'P') {
        Some(pos) => pos,
        None => return false,
    };
    let target = match layout::parsers::find_char(&grid, 'G') {
        Some(pos) => pos,
        None => return false,
    };
 
    geometry::manhattan_distance(start, target) >= min_distance as i32

}
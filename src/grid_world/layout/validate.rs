//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use crate::grid_world::layout;
use crate::grid_world::pathfinding::bfs;

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

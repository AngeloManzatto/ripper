//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use std::collections::HashSet;

use crate::grid_world::pathfinding::geometry;

//-----------------------------------------------------
// Is Blocked
//-----------------------------------------------------

pub fn is_blocked(
    grid: &Vec<Vec<char>>, 
    row: i32, 
    col: i32, 
    avoid: &Vec<char>
) -> bool
{
    
    /* ----------------------------------------------------------
    Check if a grid cell is blocked or out of bounds.

    Args:
        grid (list[list[char]]): The grid environment.
        x (usize): Row index of the cell.
        y (usize): Column index of the cell.
        tile_types_to_avoid (list[char]): List of node types to avoid.

    Returns:
        bool: True if the cell is blocked or out of bounds, False otherwise.
    ---------------------------------------------------------- */

    let height  = grid.len() as i32;
    let width = grid[0].len() as i32;

    if row < 0 || row >= height || col < 0 || col >= width { return true; }

    for a in avoid{
        if grid[row as usize][col as usize] == *a { return true};
    }
    
    false

}

//-----------------------------------------------------
// Expand neighbors
//-----------------------------------------------------

pub fn expand_neighbors(
    pos: (usize, usize),
    grid: &Vec<Vec<char>>,
    avoid: &Vec<char>,
    visited: &mut HashSet<(usize, usize)>,
) -> Vec<(usize, usize)> {

    /*
        Given `pos`, returns the walkable, in-bounds, not-yet-visited neighbors,
        marking them visited as they're returned.
     */
 
    let mut result = Vec::new();

    for (nr, nc) in geometry::neighbors(pos) {
        if is_blocked(grid, nr, nc, avoid) {
            continue;
        }

        let neighbor = (nr as usize, nc as usize);

        if visited.contains(&neighbor) {
            continue;
        }

        visited.insert(neighbor);
        result.push(neighbor);
    }

    result
}

//-----------------------------------------------------
// Test
//-----------------------------------------------------

#[cfg(test)]
    mod tests {

    use super::*;

    #[test]
    fn test_expand_neighbors() {

        let grid= vec![
            vec!['.', '#', '#', '#', '#'], 
            vec!['.', '.', '.', 'T', '#'], 
            vec!['.', '#', '#', '.', '.'], 
            vec!['.', '#', '#', '.', '#'],  
            vec!['.', '.', '.', ',', '#'],];

        let pos = (1, 1);
        let mut visited: HashSet<(usize, usize)> = HashSet::new();
        let avoid = vec!['#', 'T'];

        let result = expand_neighbors(
            pos, &grid, &avoid, &mut visited
        );

        println!("Path found {:?}", result);

        assert_eq!(result, vec![(1, 0), (1, 2)]);

    }
}



//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use std::collections::VecDeque;
use std::collections::HashSet;

use crate::grid_world::pathfinding::grid;

//-----------------------------------------------------
// Find connected regions
//-----------------------------------------------------

pub fn find_connected_regions(grid: &Vec<Vec<char>>, avoid: &Vec<char>) -> Vec<Vec<(usize, usize)>> {

    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut regions = Vec::new();

    let height = grid.len();
    let width = grid[0].len();

    for row in 0..height {
        for col in 0..width {
            let pos = (row, col);

            // If this cell was already visited or blocked skip
            if visited.contains(&pos) || grid::is_blocked(grid, row as i32, col as i32, avoid) {
                continue;
            }

            // pos is unvisited and walkable — flood-fill the whole component from here
            let mut component = Vec::new();
            let mut frontier = VecDeque::new();
            frontier.push_back(pos);
            visited.insert(pos);

            while let Some(current) = frontier.pop_front() {
                component.push(current);

                for neighbor in grid::expand_neighbors(current, grid, avoid, &mut visited) {
                    frontier.push_back(neighbor);
                }
            }

            regions.push(component);
        }
    }

    regions
}

//-----------------------------------------------------
// Get largest region
//-----------------------------------------------------

pub fn get_largest_region(regions: &Vec<Vec<(usize, usize)>>) -> Option<&Vec<(usize, usize)>> {
    regions.iter().max_by_key(|region| region.len())
}

//-----------------------------------------------------
// Test
//-----------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_connected_regions() {
        // Path find

        let grid= vec![
            vec!['.', '#', '#', '#', '#'], 
            vec!['.', '.', '.', 'T', '#'], 
            vec!['.', '#', '#', '.', '.'], 
            vec!['.', '#', '#', '.', '#'],  
            vec!['.', '.', '#', ',', '#'],];

        let avoid = vec!['#', 'T'];

        let regions = find_connected_regions(
            &grid,
            &avoid,
        );

        println!("Path found {:?}", regions);

        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0], vec![(0,0),(1,0),(2,0),(1,1),(3,0),(1,2),(4,0),(4,1)]);
        assert_eq!(regions[1], vec![(2,3),(3,3),(2,4),(4,3)]);

    }

    #[test]
    fn test_find_largest_regions() {
        // Path find

        let grid= vec![
            vec!['.', '#', '#', '#', '#'], 
            vec!['.', '.', '.', 'T', '#'], 
            vec!['.', '#', '#', '.', '.'], 
            vec!['.', '#', '#', '.', '#'],  
            vec!['.', '.', '#', ',', '#'],];

        let avoid = vec!['#', 'T'];

        let regions = find_connected_regions(
            &grid,
            &avoid,
        );

        let largest_region = get_largest_region(&regions);

        println!("Largest region {:?}", largest_region);

    }

}
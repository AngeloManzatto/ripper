//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use std::collections::VecDeque;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::grid_world::pathfinding::geometry;
use crate::grid_world::pathfinding::grid;

//-----------------------------------------------------
// Breadth First Search
//-----------------------------------------------------

pub fn bfs_search(
    grid: Vec<Vec<char>>, 
    start: (usize, usize),
    target: (usize, usize),
    avoid: Vec<char>
) -> Option<Vec<(usize, usize)>>
{
    /*--------------------------------------------------------------------------
    
    Algorithm:
    1. Push `start` onto a deque list (FIFO)
    2. Repeatedly pop the first cell on the deque list
        - If it was already visited, skip it (it can be pushed more than once).
        - Mark it visited.
        - If it is `target`, walk backward through the recorded parent links to
        rebuild the path and return it.
        - Otherwise, look at its unblocked, unvisited neighbors: record each one's
        parent (for path reconstruction) the first time it is discovered, and
        push it onto the deque.
    3. If the deque empties out without ever reaching `target`, no path exists.
    
    --------------------------------------------------------------------------*/

    let mut frontier = VecDeque::new();
    frontier.push_back(start);

    let mut visited:HashSet<(usize, usize)> = HashSet::new();
    let mut map: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let path:Vec<(usize, usize)> = Vec::new();

    // # If already at the target, do nothing
    if start == target { return Some(path); }

    while let Some(current) = frontier.pop_front() {
        
        // If we already visited this cell skip it
        if visited.contains(&current) {
            continue;
        }

        // Mark this cell as visited
        visited.insert(current);

        // If we reach the goal build result from reverse path from mapped nodes
        if current == target { 

            let mut result = vec![current];
            let mut node = current;

            while let Some(&parent) = map.get(&node) {
                result.push(parent);
                node = parent;
            }

            result.reverse();
            return Some(result);
        }

        // Check neighbors
        for (nr, nc) in geometry::neighbors(current) 
        {
            if grid::is_blocked(&grid, nr, nc, &avoid) {
                continue;
            }

            let neighbor = (nr as usize, nc as usize);

            if visited.contains(&neighbor) {
                continue;
            }

            // Only record a breadcrumb the first time we discover this cell
            map.entry(neighbor).or_insert(current);

            frontier.push_back(neighbor);
        }

    }

    
    None
}

//-----------------------------------------------------
// Test
//-----------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortest_path() {
        // Path find

        let grid= vec![
            vec!['.', '#', '#', '#', '#'], 
            vec!['.', '.', '.', '.', '#'], 
            vec!['.', '#', '#', '.', '.'], 
            vec!['.', '#', '#', '.', '#'],  
            vec!['.', '.', '.', ',', '#'],];

        let start = (0, 0);
        let target = (2, 4);
        let avoid = vec!['#', 'T'];

        let path = bfs_search(
            grid, 
            start, 
            target,
            avoid
        );

        println!("Path found {:?}", path);

    }

    #[test]
    fn test_no_path_exists() {
        // Path find

        let grid= vec![
            vec!['#', '#', '#', '#', '#'], 
            vec!['#', '.', '#', '.', '#'], 
            vec!['#', '.', '#', '.', '#'], 
            vec!['#', '#', '#', '.', '#'],  
            vec!['#', '#', '#', '#', '#'],];

        let start = (1, 1);
        let target = (3, 3);
        let avoid = vec!['#', 'T'];

        let path = bfs_search(
            grid, 
            start, 
            target,
            avoid
        );

        println!("Path found {:?}", path);

    }

    #[test]
    fn test_avoid_hazard() {
        // Path find

        let grid= vec![
            vec!['.', '#', '#', '#', '#'], 
            vec!['.', '.', '.', 'T', '#'], 
            vec!['.', '#', '#', '.', '.'], 
            vec!['.', '#', '#', '.', '#'],  
            vec!['.', '.', '.', ',', '#'],];

        let start = (0, 0);
        let target = (2, 4);
        let avoid = vec!['#', 'T'];

        let path = bfs_search(
            grid, 
            start, 
            target,
            avoid
        );

        println!("Path found {:?}", path);

    }

}
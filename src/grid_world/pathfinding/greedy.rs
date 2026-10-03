//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::grid_world::pathfinding::geometry;
use crate::grid_world::pathfinding::grid;

//-----------------------------------------------------
// Greedy Best First Search
//-----------------------------------------------------

pub fn greedy_best_first_search(
    grid: Vec<Vec<char>>,
    start: (usize, usize),
    target: (usize, usize),
    avoid: Vec<char>,
) -> Option<Vec<(usize, usize)>> {
    /*--------------------------------------------------------------------------

    Algorithm:
    1. Push `start` onto a min-heap ordered by heuristic distance to `target`.
    2. Repeatedly pop the cell with the smallest heuristic value.
        - If it was already visited, skip it (it can be pushed more than once).
        - Mark it visited.
        - If it is `target`, walk backward through the recorded parent links to
        rebuild the path, reverse it, and return it.
        - Otherwise, look at its unblocked, unvisited neighbors: record each one's
        parent (for path reconstruction) the first time it is discovered, and
        push it onto the heap.
    3. If the heap empties out without ever reaching `target`, no path exists.

    --------------------------------------------------------------------------*/

    let mut path_heap = BinaryHeap::new();
    path_heap.push((Reverse(geometry::manhattan_distance(start, target)), start));

    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut map: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let path: Vec<(usize, usize)> = Vec::new();

    // # If already at the target, do nothing
    if start == target {
        return Some(path);
    }

    while let Some((_, current)) = path_heap.pop() {
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
        for (nr, nc) in geometry::neighbors(current) {
            if grid::is_blocked(&grid, nr, nc, &avoid) {
                continue;
            }

            let neighbor = (nr as usize, nc as usize);

            if visited.contains(&neighbor) {
                continue;
            }

            // Only record a breadcrumb the first time we discover this cell
            map.entry(neighbor).or_insert(current);

            path_heap.push((
                Reverse(geometry::manhattan_distance(neighbor, target)),
                neighbor,
            ));
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

        let grid = vec![
            vec!['.', '#', '#', '#', '#'],
            vec!['.', '.', '.', '.', '#'],
            vec!['.', '#', '#', '.', '.'],
            vec!['.', '#', '#', '.', '#'],
            vec!['.', '.', '.', ',', '#'],
        ];

        let start = (0, 0);
        let target = (2, 4);
        let avoid = vec!['#', 'T'];

        let path = greedy_best_first_search(grid, start, target, avoid);

        println!("Path found {:?}", path);
    }

    #[test]
    fn test_no_path_exists() {
        // Path find

        let grid = vec![
            vec!['#', '#', '#', '#', '#'],
            vec!['#', '.', '#', '.', '#'],
            vec!['#', '.', '#', '.', '#'],
            vec!['#', '#', '#', '.', '#'],
            vec!['#', '#', '#', '#', '#'],
        ];

        let start = (1, 1);
        let target = (3, 3);
        let avoid = vec!['#', 'T'];

        let path = greedy_best_first_search(grid, start, target, avoid);

        println!("Path found {:?}", path);
    }

    #[test]
    fn test_avoid_hazard() {
        // Path find

        let grid = vec![
            vec!['.', '#', '#', '#', '#'],
            vec!['.', '.', '.', 'T', '#'],
            vec!['.', '#', '#', '.', '.'],
            vec!['.', '#', '#', '.', '#'],
            vec!['.', '.', '.', ',', '#'],
        ];

        let start = (0, 0);
        let target = (2, 4);
        let avoid = vec!['#', 'T'];

        let path = greedy_best_first_search(grid, start, target, avoid);

        println!("Path found {:?}", path);
    }
}

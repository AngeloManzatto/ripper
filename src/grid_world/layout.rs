//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use crate::grid_world::entity;
use crate::grid_world::world;
use std::collections::HashMap;
use std::collections::{VecDeque, HashSet};

use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{WeightedIndex, Distribution};

//-----------------------------------------------------
// Layout Generator Configuration
//-----------------------------------------------------
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub width: usize,
    pub height: usize,
    pub wall_density: f64,
    pub num_enemies: usize,
    pub num_traps: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        GenerationConfig {
            width: 16,
            height: 16,
            wall_density: 0.2,
            num_enemies: 1,
            num_traps: 0,
        }
    }
}

//-----------------------------------------------------
// Mutation Generator Configuration
//-----------------------------------------------------
#[derive(Clone, Debug)]
pub struct MutationConfig {
    pub cells_per_mutation: usize,
    pub max_wall_density: f64,
    pub max_enemies: u32,
    pub max_traps: u32,
    pub wall_weight: f64,
    pub enemy_weight: f64,
    pub trap_weight: f64,
}

impl Default for MutationConfig {
    fn default() -> Self {
        MutationConfig {
            cells_per_mutation: 3,
            max_wall_density: 0.3,
            max_enemies: 3,
            max_traps: 3,
            wall_weight: 0.80,
            enemy_weight: 0.10,
            trap_weight: 0.10,
        }
    }
}

enum MutationCategory {
    Wall,
    Enemy,
    Trap,
}

//-----------------------------------------------------
// Generate Layout
//-----------------------------------------------------

pub fn generate_layout(config: &GenerationConfig) -> String {
    let width = config.width;
    let height = config.height;

    // 1. Build a height x width grid of chars, all '.' initially
    let mut grid: Vec<Vec<char>> = vec![vec!['.'; width]; height];

    // 2. Fill the border with '#' (row 0, row height-1, col 0, col width-1)
    let mut empty_positions = Vec::<(usize, usize)>::new();
    for row in 0..height  {
        for col in 0..width {

            if row == 0 || row == height - 1 || col == 0 || col == width - 1{
                grid[row][col] = '#';
            } else{
                empty_positions.push((row, col));
            }

        }
    };

    // 3. For every interior cell (not border), randomly place '#' based on config.wall_density
    let mut rng = rand::thread_rng();

    for row in 1..height - 1 {
        for col in 1..width - 1 {
            if rng.gen_bool(config.wall_density) {
                grid[row][col] = '#';
            }
        }
    }

    // 4. Place P, G, num_enemies E's, and num_traps T's at random EMPTY interior positions
    //    (need a helper: find a random position where grid[row][col] == '.', then overwrite it)
    //    your code here
    let n_elements = 2 + config.num_enemies + config.num_traps; // Player + Goal + Enemies + Traps
    let mut selected_positions: Vec<_> = empty_positions.choose_multiple(&mut rng, n_elements).collect();

    let player_position = selected_positions.pop().expect("Position not found");
    grid[player_position.0][player_position.1] = 'P';

    let goal_position = selected_positions.pop().expect("Position not found");
    grid[goal_position.0][goal_position.1] = 'G';

    for _n in  0..config.num_enemies{
        let enemy_position = selected_positions.pop().expect("Position not found");
        grid[enemy_position.0][enemy_position.1] = 'E';
    }

    for _n in  0..config.num_traps{
        let trap_position = selected_positions.pop().expect("Position not found");
        grid[trap_position.0][trap_position.1] = 'T';
    }

    // 5. Join the grid into a single newline-separated String
    grid.iter()
    .map(|row| row.iter().collect::<String>())
    .collect::<Vec<String>>()
    .join("\n")

}

//-----------------------------------------------------
// Parse Layout
//-----------------------------------------------------

pub fn parse_layout(layout: &str, max_tick:u32) -> world::World {

    // Initialize string of lines iterator
    let lines: Vec<&str> = layout.lines().collect();

    // Get grid size
    let height = lines.len();
    let width = lines[0].trim().len();

    // Initalize world
    let mut world = world::World {
        width,
        height,
        next_id: 0,
        positions: std::collections::HashMap::new(),
        kinds: std::collections::HashMap::new(),
        layout: layout.to_string(),
        done: false,
        end_reason: None,
        tick:0,
        max_tick: max_tick
    };

    for (row, line) in lines.iter().enumerate() {

        let trimmed_line = &line.trim();

        if trimmed_line.len() != width {
            panic!("Ragged layout: row {} has length {}, expected {}", row, line.len(), width);
        }

        for (col, ch) in trimmed_line.chars().enumerate() {
            let kind = match ch {
                '#' => Some(entity::EntityKind::Wall),
                'P' => Some(entity::EntityKind::Player),
                'E' => Some(entity::EntityKind::Enemy),
                'G' => Some(entity::EntityKind::Goal),
                'T' => Some(entity::EntityKind::Trap),
                '.' => None,
                _ => panic!("Unknown layout character: '{}'", ch),
            };

            if let Some(k) = kind {
                world::spawn_entity(&mut world, k, (row, col));
            }
        }
    }

    world
}

//-----------------------------------------------------
// Layout to Chars
//-----------------------------------------------------

fn layout_to_chars(layout: &str) -> Vec<Vec<char>> {
    layout.lines().map(|line| line.chars().collect()).collect()
}

//-----------------------------------------------------
// Find Char
//-----------------------------------------------------

fn find_char(grid: &[Vec<char>], target: char) -> Option<(usize, usize)> {
    for (row, line) in grid.iter().enumerate() {
        for (col, &ch) in line.iter().enumerate() {
            if ch == target {
                return Some((row, col));
            }
        }
    }
    None
}

//-----------------------------------------------------
// BFS for reachability
//-----------------------------------------------------

pub fn is_layout_reachable(layout: &str) -> bool {

    let grid = layout_to_chars(layout);
    let height = grid.len();
    let width = grid[0].len();

    let start = find_char(&grid, 'P').expect("no P found in layout");
    let target = find_char(&grid, 'G').expect("no G found in layout");

    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

     // The four possible moves from any cell: up, down, left, right.
    let deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    // Seed the search: start is the first cell we know about.
    queue.push_back(start);
    visited.insert(start);

    // Keep exploring as long as there are undiscovered cells left to check.
    // `pop_front()` takes the OLDEST cell still in the queue — this is what
    // guarantees we explore in order of increasing distance from `start`.
    while let Some(current) = queue.pop_front() {

        // Found it — there's a walkable path from start to target.
        if current == target {
            return true;
        }

        // Look at all four neighbors of the current cell.
        for (dr, dc) in deltas {
            let new_row = current.0 as i32 + dr;
            let new_col = current.1 as i32 + dc;

            // Skip neighbors that would fall outside the grid entirely.
            // (Signed arithmetic here avoids the usize-underflow trap —
            // see clamp_move for the same pattern.)
            if new_row < 0 || new_row >= height as i32 || new_col < 0 || new_col >= width as i32 {
                continue;
            }

            let neighbor = (new_row as usize, new_col as usize);

            // Only explore this neighbor if it's not blocked by a wall,
            // AND we haven't already discovered it before (avoids
            // re-processing the same cell, and avoids infinite loops).
            if grid[neighbor.0][neighbor.1] != '#' && !visited.contains(&neighbor) {
                visited.insert(neighbor);   // mark as discovered
                queue.push_back(neighbor);  // and schedule it for exploration
            }
        }
    }

    false
}


//-----------------------------------------------------
// Generate Valid Layout
//-----------------------------------------------------

pub fn generate_valid_layout(config: &GenerationConfig, max_attempts: u32) -> String {
    for _ in 0..max_attempts {
        let layout_str = generate_layout(config);
        if is_layout_reachable(&layout_str) {
            return layout_str;
        }
    }
    panic!("Could not produce a valid layout after {} attempts", max_attempts);
}

//-----------------------------------------------------
// Mutate Layout
//-----------------------------------------------------

pub fn mutate_layout(layout: &str, config: &MutationConfig) -> String {
    let mut grid = layout_to_chars(layout);
    let height = grid.len();
    let width = grid[0].len();

    let mut map_positions: HashMap<char, Vec<(usize, usize)>> = HashMap::new();

    for row in 1..height - 1 {
        for col in 1..width - 1 {
            let ch = grid[row][col];
            map_positions.entry(ch).or_insert_with(Vec::new).push((row, col));
        }
    }

    let interior_cells = (height - 2) * (width - 2);
    let mut wall_count = map_positions.get(&'#').map(|v| v.len()).unwrap_or(0);

    let weights = [config.wall_weight, config.enemy_weight, config.trap_weight];
    let dist = WeightedIndex::new(&weights).expect("invalid weights");
    let mut rng = rand::thread_rng();

    for _ in 0..config.cells_per_mutation {

        let category = match dist.sample(&mut rng) {
            0 => MutationCategory::Wall,
            1 => MutationCategory::Enemy,
            2 => MutationCategory::Trap,
            _ => unreachable!(),
        };

        match category {
            MutationCategory::Wall => {
                let mut empty_positions = map_positions.remove(&'.').unwrap_or_default();
                let mut wall_positions = map_positions.remove(&'#').unwrap_or_default();

                if !empty_positions.is_empty() && (wall_count + 1) as f64 / interior_cells as f64 <= config.max_wall_density {
                    let idx = rng.gen_range(0..empty_positions.len());
                    let pos = empty_positions.remove(idx);
                    grid[pos.0][pos.1] = '#';
                    wall_positions.push(pos);
                    wall_count += 1;
                } else if !wall_positions.is_empty() {
                    let idx = rng.gen_range(0..wall_positions.len());
                    let pos = wall_positions.remove(idx);
                    grid[pos.0][pos.1] = '.';
                    empty_positions.push(pos);
                    wall_count -= 1;
                }

                map_positions.insert('.', empty_positions);
                map_positions.insert('#', wall_positions);
            }
            MutationCategory::Enemy => {
                let mut empty_positions = map_positions.remove(&'.').unwrap_or_default();
                let mut enemy_positions = map_positions.remove(&'E').unwrap_or_default();
                let enemy_count = enemy_positions.len() as u32;

                if !empty_positions.is_empty() && enemy_count < config.max_enemies {
                    let idx = rng.gen_range(0..empty_positions.len());
                    let pos = empty_positions.remove(idx);
                    enemy_positions.push(pos);
                    grid[pos.0][pos.1] = 'E';
                }

                map_positions.insert('.', empty_positions);
                map_positions.insert('E', enemy_positions);
            }
            MutationCategory::Trap => {
                let mut empty_positions = map_positions.remove(&'.').unwrap_or_default();
                let mut trap_positions = map_positions.remove(&'T').unwrap_or_default();
                let trap_count = trap_positions.len() as u32;

                if !empty_positions.is_empty() && trap_count < config.max_traps {
                    let idx = rng.gen_range(0..empty_positions.len());
                    let pos = empty_positions.remove(idx);
                    trap_positions.push(pos);
                    grid[pos.0][pos.1] = 'T';
                }

                map_positions.insert('.', empty_positions);
                map_positions.insert('T', trap_positions);
            }
        }
    }

    grid.iter()
        .map(|row| row.iter().collect::<String>())
        .collect::<Vec<String>>()
        .join("\n")
}

//-----------------------------------------------------
// Mutate Valid Layout
//-----------------------------------------------------
pub fn mutate_valid_layout(layout: &str, config: &MutationConfig, max_attempts: u32) -> String {
    for _ in 0..max_attempts {
        let candidate = mutate_layout(layout, config);
        if is_layout_reachable(&candidate) {
            return candidate;
        }
    }
    panic!("Could not produce a valid mutation after {} attempts", max_attempts);
}
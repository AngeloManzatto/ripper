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
    // Wall mutation
    pub n_wall_mutations: usize,
    pub max_wall_density: f64,

    // Enemy mutation (independent Add/Remove/Nothing weights)
    pub max_enemies: u32,
    pub enemy_add_weight: f64,
    pub enemy_remove_weight: f64,
    pub enemy_nothing_weight: f64,

    // Trap mutation (independent Add/Remove/Nothing weights)
    pub max_traps: u32,
    pub trap_add_weight: f64,
    pub trap_remove_weight: f64,
    pub trap_nothing_weight: f64,

    // Reposition mutation
    pub n_repositions: usize,
}

impl Default for MutationConfig {
    fn default() -> Self {
        MutationConfig {
            n_wall_mutations: 3,
            max_wall_density: 0.3,

            max_enemies: 3,
            enemy_add_weight: 0.4,
            enemy_remove_weight: 0.2,
            enemy_nothing_weight: 0.4,

            max_traps: 3,
            trap_add_weight: 0.2,
            trap_remove_weight: 0.1,
            trap_nothing_weight: 0.7,

            n_repositions: 1,
        }
    }
}

enum MutationType {
    Add,
    Remove,
    Nothing,
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

fn mutate_walls(
    grid: &mut Vec<Vec<char>>, 
    map_positions: &mut HashMap<char, Vec<(usize, usize)>>,
    config: &MutationConfig
)
{
    let mut rng = rand::thread_rng();

    let height = grid.len();
    let width = grid[0].len();

    let interior_cells = (height - 2) * (width - 2);
    let mut wall_count = map_positions
                                .get(&'#')
                                .map(|v| v.len()).unwrap_or(0);
    
    for _n in 0..config.n_wall_mutations {

        // Borrow vectors to mutation from hashmap
        let mut empty_positions = map_positions.remove(&'.').unwrap_or_default();
        let mut wall_positions = map_positions.remove(&'#').unwrap_or_default();
        let curr_wall_density = (wall_count + 1) as f64 / interior_cells as f64;

        // If we still have empty spaces and current wall density is below limit change it to wall
        if !empty_positions.is_empty() && curr_wall_density <= config.max_wall_density {

            // Get a random emtpy position
            let idx = rng.gen_range(0..empty_positions.len());
            let pos = empty_positions.remove(idx);

            // Turn it into wall
            grid[pos.0][pos.1] = '#';

            // Update wall position vector and wall density count
            wall_positions.push(pos);
            wall_count += 1;

        } else if !wall_positions.is_empty() {

            // Get a random wall position
            let idx = rng.gen_range(0..wall_positions.len());
            let pos = wall_positions.remove(idx);

            // Turn it into an empty space
            grid[pos.0][pos.1] = '.';

            // Update empty vector and wall density count 
            empty_positions.push(pos);
            wall_count -= 1;
        }

        // Return vectors to hashmap
        map_positions.insert('.', empty_positions);
        map_positions.insert('#', wall_positions);
        
    }

}

fn roll_mutation_type(add_weight: f64, remove_weight: f64, nothing_weight: f64) -> MutationType {
    let weights = [add_weight, remove_weight, nothing_weight];
    let dist = WeightedIndex::new(&weights).expect("invalid weights");
    let mut rng = rand::thread_rng();

    match dist.sample(&mut rng) {
        0 => MutationType::Add,
        1 => MutationType::Remove,
        _ => MutationType::Nothing,
    }
}

fn mutate_entity(
    grid: &mut Vec<Vec<char>>, 
    map_positions: &mut HashMap<char, Vec<(usize, usize)>>,
    entity_type: char, 
    max_count: usize,
    mutation_type: MutationType
)
{
    let mut rng = rand::thread_rng();

    let n_entities = map_positions.get(&entity_type)
                                         .map(|v| v.len())
                                         .unwrap_or(0);

    // Borrow vectors to mutation from hashmap
    let mut empty_positions = map_positions.remove(&'.').unwrap_or_default();
    let mut entity_positions = map_positions.remove(&entity_type).unwrap_or_default();

    match mutation_type {
        MutationType::Add => {
            if n_entities < max_count && !empty_positions.is_empty() {

                // Get a random emtpy position
                let idx = rng.gen_range(0..empty_positions.len());
                let pos = empty_positions.remove(idx);

                // Turn it into entity
                grid[pos.0][pos.1] = entity_type;

                // Update entity vector
                entity_positions.push(pos);

            }
        }
        MutationType::Remove => {
            if n_entities > 0 {
                let idx = rng.gen_range(0..entity_positions.len());
                let pos = entity_positions.remove(idx);

                // Turn it into empty
                grid[pos.0][pos.1] = '.';

                // Update entity vector
                empty_positions.push(pos);
            }
        }
        MutationType::Nothing => {

        }
        
    }

    // Return vectors to hashmap
    map_positions.insert('.', empty_positions);
    map_positions.insert(entity_type, entity_positions);

}

fn mutate_repositions(
    grid: &mut Vec<Vec<char>>,
    map_positions: &mut HashMap<char, Vec<(usize, usize)>>,
    n_repositions: usize,
)
{
    let entity_chars = ['P', 'G', 'E', 'T'];
    let mut all_entities: Vec<(char, (usize, usize))> = Vec::new();

    for &ch in &entity_chars {
        if let Some(positions) = map_positions.get(&ch) {
            for &pos in positions {
                all_entities.push((ch, pos));
            }
        }
    }

    let mut rng = rand::thread_rng();
    let mut selected_positions: Vec<_> = all_entities
                                                                  .choose_multiple(&mut rng, n_repositions)
                                                                  .collect();

    for _ in 0..n_repositions {
        let mut empty_positions = map_positions.remove(&'.').unwrap_or_default();

        let (entity_type, old_pos) = selected_positions.pop().expect("Position not found");

        let idx = rng.gen_range(0..empty_positions.len());
        let new_pos = empty_positions.remove(idx);

        grid[old_pos.0][old_pos.1] = '.';
        grid[new_pos.0][new_pos.1] = *entity_type;

        // Update the entity's tracked position inside its own list
        if let Some(entity_positions) = map_positions.get_mut(&entity_type) {
            if let Some(p) = entity_positions.iter_mut().find(|p| **p == *old_pos) {
                *p = new_pos;
            }
        }

        map_positions.insert('.', empty_positions);
    }

}

pub fn mutate_layout(layout: &str, config: &MutationConfig) -> String {

    // Turn the layout into a grid of chars
    let mut grid = layout_to_chars(layout);
    let height = grid.len();
    let width = grid[0].len();

    // get mapping between chars types and positions
    let mut map_positions: HashMap<char, Vec<(usize, usize)>> = HashMap::new();

    for row in 1..height - 1 {
        for col in 1..width - 1 {
            let ch = grid[row][col];
            map_positions.entry(ch).or_insert_with(Vec::new).push((row, col));
        }
    }

    //--------------------------------------------------
    // Mutate walls
    //--------------------------------------------------
    mutate_walls(&mut grid, &mut map_positions, &config);

    //--------------------------------------------------
    // Mutate enemies
    //--------------------------------------------------
    let enemy_mutation = roll_mutation_type(
        config.enemy_add_weight, 
        config.enemy_remove_weight, 
        config.enemy_nothing_weight);

    mutate_entity(
        &mut grid, 
        &mut map_positions, 
        'E', 
        config.max_enemies as usize, 
        enemy_mutation);

    //--------------------------------------------------
    // Mutate traps
    //--------------------------------------------------
    let trap_mutation = roll_mutation_type(
        config.trap_add_weight, 
        config.trap_remove_weight, 
        config.trap_nothing_weight);

    mutate_entity(
        &mut grid, 
        &mut map_positions, 
        'T', 
        config.max_traps as usize, 
        trap_mutation);

    //--------------------------------------------------
    // Mutate reposition
    //--------------------------------------------------     
    mutate_repositions(&mut grid, &mut map_positions, config.n_repositions);    

    //--------------------------------------------------
    // Convert mutated grid into string again
    //--------------------------------------------------
    grid.iter()
        .map(|row| row.iter().collect::<String>())
        .collect::<Vec<String>>()
        .join("\n")
}

/*

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
            MutationCategory::Reposition => {
                let mut empty_positions = map_positions.remove(&'.').unwrap_or_default();

                // Gather all entity characters currently present, each with its position
                let entity_chars = ['P', 'G', 'E', 'T'];
                let mut all_entities: Vec<(char, (usize, usize))> = Vec::new();
                for &ch in &entity_chars {
                    if let Some(positions) = map_positions.get(&ch) {
                        for &pos in positions {
                            all_entities.push((ch, pos));
                        }
                    }
                }

                if !all_entities.is_empty() && !empty_positions.is_empty() {
                    let idx = rng.gen_range(0..all_entities.len());
                    let (entity_char, old_pos) = all_entities[idx];

                    let new_idx = rng.gen_range(0..empty_positions.len());
                    let new_pos = empty_positions.remove(new_idx);

                    // Move: clear old cell, place entity at new cell
                    grid[old_pos.0][old_pos.1] = '.';
                    grid[new_pos.0][new_pos.1] = entity_char;
                    empty_positions.push(old_pos);

                    // Update this entity's specific position within its own list
                    if let Some(positions) = map_positions.get_mut(&entity_char) {
                        if let Some(p) = positions.iter_mut().find(|p| **p == old_pos) {
                            *p = new_pos;
                        }
                    }
                }

                map_positions.insert('.', empty_positions);
            }
        }
    }

    grid.iter()
        .map(|row| row.iter().collect::<String>())
        .collect::<Vec<String>>()
        .join("\n")
}
*/
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
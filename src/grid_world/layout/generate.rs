//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use rand::Rng;
use rand::seq::SliceRandom;

use crate::grid_world::layout::{parsers, validate};
use crate::grid_world::pathfinding::regions;

//-----------------------------------------------------
// Generation
//-----------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct GenerationConfig {
    pub width: usize,
    pub height: usize,
    pub wall_density: f64,
    pub num_enemies: usize,
    pub num_traps: usize,
    pub min_player_goal_distance: usize,
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        GenerationConfig {
            width: 16,
            height: 16,
            wall_density: 0.2,
            num_enemies: 1,
            num_traps: 0,
            min_player_goal_distance: 0,
            seed: Some(0),
        }
    }
}

//-----------------------------------------------------
// Generate a default grid
//-----------------------------------------------------

fn generate_default_grid(width: usize, height: usize) -> Vec<Vec<char>> {
    vec![vec!['.'; width]; height]
}

//-----------------------------------------------------
// Pad grid with walls
//-----------------------------------------------------

fn pad_grid_with_walls(mut grid: Vec<Vec<char>>, padding: usize) -> Vec<Vec<char>> {
    let height = grid.len();
    let width = grid[0].len();

    // Padding can't consume more than half of either dimension,
    // or the whole grid becomes wall.
    let max_padding = (height.min(width).saturating_sub(1)) / 2;
    let padding = padding.min(max_padding);

    for row in 0..height {
        for col in 0..width {
            if row < padding || row >= height - padding || col < padding || col >= width - padding {
                grid[row][col] = '#';
            }
        }
    }

    grid
}

//-----------------------------------------------------
// Add grid with walls
//-----------------------------------------------------

fn add_random_walls_to_grid(
    mut grid: Vec<Vec<char>>,
    wall_density: f64,
    rng: &mut impl Rng,
) -> Vec<Vec<char>> {
    let height = grid.len();
    let width = grid[0].len();

    for row in 0..height - 1 {
        for col in 0..width - 1 {
            if rng.gen_bool(wall_density) {
                grid[row][col] = '#';
            }
        }
    }

    grid
}

//-----------------------------------------------------
// Get empty positions
//-----------------------------------------------------

fn get_grid_empty_cells(grid: &Vec<Vec<char>>) -> Vec<(usize, usize)> {
    let height = grid.len();
    let width = grid[0].len();

    // Get valid positions that are not occupied
    let mut empty_cells = Vec::new();

    for row in 0..height {
        for col in 0..width {
            if grid[row][col] == '.' {
                empty_cells.push((row, col));
            }
        }
    }

    empty_cells
}

//-----------------------------------------------------
// Pick player and goal position
//-----------------------------------------------------

fn pick_player_and_goal(
    empty_cells: &[(usize, usize)],
    min_distance: usize,
    rng: &mut impl Rng,
) -> Option<((usize, usize), (usize, usize))> {
    if empty_cells.len() < 2 {
        return None;
    }

    for _ in 0..100 {
        let mut candidates = empty_cells.choose_multiple(rng, 2);
        let player_pos = *candidates.next().unwrap();
        let goal_pos = *candidates.next().unwrap();

        let dist = player_pos.0.abs_diff(goal_pos.0) + player_pos.1.abs_diff(goal_pos.1);
        if dist >= min_distance {
            return Some((player_pos, goal_pos));
        }
    }

    None
}

//-----------------------------------------------------
// Generate a default layout
//-----------------------------------------------------

pub fn generate_layout(config: &GenerationConfig, rng: &mut impl Rng) -> Option<String> {
    /*--------------------------------------------------------
    Generation procedure:

    1) Generate a grid of (height x width) size with emtpy cells ("." symbol).
    2) Pad the grid with walls ("#" symbol)
    3) Add walls based on config density
    4) Place traps
    5) Find connected regions
    6) Place Player and Goal on largest region
    7) Place enemies from the same region, excluding whatever cells player/goal now occupy.
    8) Serialize to String.

    --------------------------------------------------------*/

    let width = config.width;
    let height = config.height;

    // 1) Generate a grid of (height x width) size with emtpy cells ("." symbol).
    let mut grid = generate_default_grid(width, height);

    // 2) Pad the grid with walls ("#" symbol)
    grid = pad_grid_with_walls(grid, 1);

    // 3) Add walls based on config density
    grid = add_random_walls_to_grid(grid, config.wall_density, rng);

    // 4) Place traps
    let empty_cells = get_grid_empty_cells(&grid);

    let mut trap_positions: Vec<_> = empty_cells.choose_multiple(rng, config.num_traps).collect();

    for _n in 0..config.num_traps {
        let trap_position = trap_positions.pop().expect("Position not found");
        grid[trap_position.0][trap_position.1] = 'T';
    }

    // 5) Find connected regions
    let regions = regions::find_connected_regions(&grid, &vec!['#', 'T']);

    // 6) Place Player and Goal on largest region
    let largest_region = regions::get_largest_region(&regions)?;

    let (player_pos, goal_pos) =
        pick_player_and_goal(largest_region, config.min_player_goal_distance, rng)?;

    grid[player_pos.0][player_pos.1] = 'P';
    grid[goal_pos.0][goal_pos.1] = 'G';

    // 7) Place enemies from the same region, excluding whatever cells player/goal now occupy.
    let candidate_cells: Vec<(usize, usize)> = largest_region
        .iter()
        .filter(|c| **c != player_pos && **c != goal_pos)
        .cloned()
        .collect();

    let mut enemy_positions: Vec<_> = candidate_cells
        .choose_multiple(rng, config.num_enemies)
        .collect();

    for _n in 0..config.num_enemies {
        let enemy_position = enemy_positions.pop().expect("Position not found");
        grid[enemy_position.0][enemy_position.1] = 'E';
    }

    // 8) Serialize to String.
    let layout_string = parsers::grid_to_layout(&grid);

    Some(layout_string)
}

//-----------------------------------------------------
// Generate valid layout
//-----------------------------------------------------

pub fn generate_valid_layout(config: &GenerationConfig, rng: &mut impl Rng) -> Option<String> {
    const MAX_ATTEMPTS: usize = 20;

    for _attempt in 0..MAX_ATTEMPTS {
        if let Some(layout) = generate_layout(config, rng) {
            if validate::is_layout_reachable(&layout)
                && validate::is_player_far_from_goal(&layout, config.min_player_goal_distance)
            {
                return Some(layout);
            }
        }
    }

    None // exhausted retries
}
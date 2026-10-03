//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use rand::Rng;
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;

use std::collections::HashMap;

use crate::grid_world::layout::parsers;

//-----------------------------------------------------
// Mutation Generator Configuration
//-----------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct MutationConfig {
    // Wall mutation
    pub n_wall_mutations: usize,
    pub max_wall_density: f64,

    // Enemy mutation (independent Add/Remove/Nothing weights)
    pub max_enemies: usize,
    pub enemy_add_weight: f64,
    pub enemy_remove_weight: f64,
    pub enemy_nothing_weight: f64,

    // Trap mutation (independent Add/Remove/Nothing weights)
    pub max_traps: usize,
    pub trap_add_weight: f64,
    pub trap_remove_weight: f64,
    pub trap_nothing_weight: f64,

    // Reposition mutation
    pub n_repositions: usize,
    pub min_player_goal_distance: usize,

    pub seed: Option<u64>,
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
            min_player_goal_distance: 0,

            seed: Some(0),
        }
    }
}

//-----------------------------------------------------
// Mutate walls
//-----------------------------------------------------

fn mutate_walls(
    grid: &mut Vec<Vec<char>>,
    map_positions: &mut HashMap<char, Vec<(usize, usize)>>,
    config: &MutationConfig,
    rng: &mut impl Rng,
) {
    let height = grid.len();
    let width = grid[0].len();

    let interior_cells = (height - 2) * (width - 2);
    let mut wall_count = map_positions.get(&'#').map(|v| v.len()).unwrap_or(0);

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

//-----------------------------------------------------
// Mutate entitiy
//-----------------------------------------------------

enum MutationType {
    Add,
    Remove,
    Nothing,
}

fn roll_mutation_type(
    add_weight: f64,
    remove_weight: f64,
    nothing_weight: f64,
    rng: &mut impl Rng,
) -> MutationType {
    let weights = [add_weight, remove_weight, nothing_weight];
    let dist = WeightedIndex::new(&weights).expect("invalid weights");

    match dist.sample(rng) {
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
    mutation_type: MutationType,
    rng: &mut impl Rng,
) {
    let n_entities = map_positions
        .get(&entity_type)
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
        MutationType::Nothing => {}
    }

    // Return vectors to hashmap
    map_positions.insert('.', empty_positions);
    map_positions.insert(entity_type, entity_positions);
}

//-----------------------------------------------------
// Mutate reposition
//-----------------------------------------------------

fn mutate_repositions(
    grid: &mut Vec<Vec<char>>,
    map_positions: &mut HashMap<char, Vec<(usize, usize)>>,
    n_repositions: usize,
    rng: &mut impl Rng,
) -> Option<()> {
    let entity_chars = ['P', 'G', 'E', 'T'];
    let mut all_entities: Vec<(char, (usize, usize))> = Vec::new();

    for &ch in &entity_chars {
        if let Some(positions) = map_positions.get(&ch) {
            for &pos in positions {
                all_entities.push((ch, pos));
            }
        }
    }

    let n = n_repositions.min(all_entities.len()); // clamp instead of trusting the caller
    let mut selected_positions: Vec<_> = all_entities.choose_multiple(rng, n).collect();

    for _ in 0..n {
        let mut empty_positions = map_positions.remove(&'.').unwrap_or_default();
        if empty_positions.is_empty() {
            map_positions.insert('.', empty_positions);
            return None; // nowhere left to reposition into
        }

        let (entity_type, old_pos) = selected_positions.pop()?;
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

    Some(())
}

//-----------------------------------------------------
// Mutate Layout
//-----------------------------------------------------

pub fn mutate_layout(layout: &str, config: &MutationConfig, rng: &mut impl Rng) -> Option<String> {
    // Turn the layout into a grid of chars
    let mut grid = parsers::layout_to_grid(layout);
    let height = grid.len();
    let width = grid[0].len();

    // get mapping between chars types and positions
    let mut map_positions: HashMap<char, Vec<(usize, usize)>> = HashMap::new();

    for row in 1..height - 1 {
        for col in 1..width - 1 {
            let ch = grid[row][col];
            map_positions
                .entry(ch)
                .or_insert_with(Vec::new)
                .push((row, col));
        }
    }

    //--------------------------------------------------
    // Mutate walls
    //--------------------------------------------------

    mutate_walls(&mut grid, &mut map_positions, config, rng);

    //--------------------------------------------------
    // Mutate enemies
    //--------------------------------------------------
    let enemy_mutation = roll_mutation_type(
        config.enemy_add_weight,
        config.enemy_remove_weight,
        config.enemy_nothing_weight,
        rng,
    );

    mutate_entity(
        &mut grid,
        &mut map_positions,
        'E',
        config.max_enemies as usize,
        enemy_mutation,
        rng,
    );

    //--------------------------------------------------
    // Mutate traps
    //--------------------------------------------------
    let trap_mutation = roll_mutation_type(
        config.trap_add_weight,
        config.trap_remove_weight,
        config.trap_nothing_weight,
        rng,
    );

    mutate_entity(
        &mut grid,
        &mut map_positions,
        'T',
        config.max_traps as usize,
        trap_mutation,
        rng,
    );

    //--------------------------------------------------
    // Mutate reposition
    //--------------------------------------------------
    mutate_repositions(&mut grid, &mut map_positions, config.n_repositions, rng)?;

    //--------------------------------------------------
    // Convert mutated grid into string again
    //--------------------------------------------------
    let layout_string = parsers::grid_to_layout(&grid);

    Some(layout_string)
}

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_world::layout::{parsers, validate};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn sample_layout() -> &'static str {
        "##########\n\
         #........#\n\
         #..E...T.#\n\
         #....P...#\n\
         #........#\n\
         #.......G#\n\
         #........#\n\
         ##########"
    }

    #[test]
    fn test_mutate_layout_deterministic() {
        let config = MutationConfig::default();

        let mut rng1 = StdRng::seed_from_u64(42);
        let result1 =
            mutate_layout(sample_layout(), &config, &mut rng1).expect("mutation should succeed");

        let mut rng2 = StdRng::seed_from_u64(42);
        let result2 =
            mutate_layout(sample_layout(), &config, &mut rng2).expect("mutation should succeed");

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_mutate_layout_preserves_dimensions() {
        let config = MutationConfig::default();
        let mut rng = StdRng::seed_from_u64(7);

        let mutated =
            mutate_layout(sample_layout(), &config, &mut rng).expect("mutation should succeed");
        let grid = parsers::layout_to_grid(&mutated);
        let original = parsers::layout_to_grid(sample_layout());

        assert_eq!(grid.len(), original.len());
        assert_eq!(grid[0].len(), original[0].len());
    }

    #[test]
    fn test_mutate_layout_keeps_single_player_and_goal() {
        let config = MutationConfig::default();
        let mut rng = StdRng::seed_from_u64(7);

        let mutated =
            mutate_layout(sample_layout(), &config, &mut rng).expect("mutation should succeed");
        let grid = parsers::layout_to_grid(&mutated);

        let p_count = grid.iter().flatten().filter(|&&c| c == 'P').count();
        let g_count = grid.iter().flatten().filter(|&&c| c == 'G').count();

        assert_eq!(p_count, 1, "expected exactly one Player");
        assert_eq!(g_count, 1, "expected exactly one Goal");
    }

    #[test]
    fn test_mutate_layout_border_untouched() {
        let config = MutationConfig::default();
        let mut rng = StdRng::seed_from_u64(7);

        let mutated =
            mutate_layout(sample_layout(), &config, &mut rng).expect("mutation should succeed");
        let grid = parsers::layout_to_grid(&mutated);
        let (height, width) = (grid.len(), grid[0].len());

        for col in 0..width {
            assert_eq!(grid[0][col], '#');
            assert_eq!(grid[height - 1][col], '#');
        }
        for row in 0..height {
            assert_eq!(grid[row][0], '#');
            assert_eq!(grid[row][width - 1], '#');
        }
    }

    #[test]
    fn test_mutate_layout_reachability_is_not_yet_guaranteed() {
        // mutate_layout can disconnect the goal (e.g. a wall toggle sealing
        // off a region) -- it does NOT retry on failure. This test just
        // documents current behavior; mutate_valid_layout is what will
        // enforce reachability via retries.
        let config = MutationConfig::default();
        let mut rng = StdRng::seed_from_u64(7);

        let mutated =
            mutate_layout(sample_layout(), &config, &mut rng).expect("mutation should succeed");

        println!("reachable: {}", validate::is_layout_reachable(&mutated));
    }
}

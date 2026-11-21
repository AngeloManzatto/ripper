//-----------------------------------------------------
// Golden test: the engine's outputs for fixed seeds
//-----------------------------------------------------

/*  A refactor must not change what the engine produces. This test runs the generator, the
mutator and the per-game repositioning for fixed seeds, writes everything that matters into one
text (layouts, positions, the player's observation grid with its cell codes, and how often each
step fails) and compares it with a file saved earlier.

First time, BEFORE the refactor, on the current engine:   UPDATE_GOLDEN=1 cargo test golden
Every time after:                                          cargo test golden

When a change is MEANT to alter the outputs (the single-region rule), run it with UPDATE_GOLDEN=1
again and read the diff of golden/engine.txt: it is the exact list of what changed.
*/

use std::fmt::Write as _;
use std::fs;

use rand::{rngs::StdRng, SeedableRng};

use crate::grid_world::entity::TileKind;
use crate::grid_world::layout::generate::{generate_valid_layout, GenerationConfig};
use crate::grid_world::layout::mutate::{mutate_valid_layout, MutationConfig};
use crate::grid_world::world::state::{World, WorldConfig};

const GOLDEN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/golden");
const GOLDEN_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/golden/engine.txt");
const N_SEEDS: u64 = 200;
const N_MUTATION_STEPS: usize = 10;

const SAMPLE_LAYOUT: &str = "##########\n\
                             #........#\n\
                             #..E...T.#\n\
                             #....P...#\n\
                             #........#\n\
                             #.......G#\n\
                             #........#\n\
                             ##########";

//-----------------------------------------------------
// The fingerprint
//-----------------------------------------------------

fn fingerprint() -> String {
    
    let mut out = String::new();

    // 1) Generation, from sparse to dense
    for &density in &[0.1, 0.2, 0.4, 0.55] {
        let config = GenerationConfig {
            width: 16,
            height: 16,
            wall_density: density,
            num_enemies: 1,
            num_traps: 2,
            min_player_goal_distance: 0,
            seed: None,
        };
        let mut failures = 0;
        for seed in 0..N_SEEDS {
            let mut rng = StdRng::seed_from_u64(seed);
            match generate_valid_layout(&config, &mut rng) {
                Some(layout) => writeln!(out, "generate density={density} seed={seed}\n{layout}").unwrap(),
                None => {
                    failures += 1;
                    writeln!(out, "generate density={density} seed={seed}\nNONE").unwrap();
                }
            }
        }
        writeln!(out, "SUMMARY generate density={density}: {failures}/{N_SEEDS} returned None").unwrap();
    }

    // 2) Mutation: a chain of mutations per seed, so the layouts get denser
    for &max_wall_density in &[0.3, 0.6] {
        let config = MutationConfig { max_wall_density, ..MutationConfig::default() };
        let mut failures = 0;
        for seed in 0..N_SEEDS {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut layout = SAMPLE_LAYOUT.to_string();
            let mut stopped_at = None;
            for step in 0..N_MUTATION_STEPS {
                match mutate_valid_layout(&layout, &config, &mut rng) {
                    Some(next) => layout = next,
                    None => {
                        stopped_at = Some(step);
                        break;
                    }
                }
            }
            if stopped_at.is_some() {
                failures += 1;
            }
            writeln!(out, "mutate cap={max_wall_density} seed={seed} stopped_at={stopped_at:?}\n{layout}").unwrap();
        }
        writeln!(out, "SUMMARY mutate cap={max_wall_density}: {failures}/{N_SEEDS} chains stopped early").unwrap();
    }

    // 3) Per-game repositioning and what the player observes (pins the cell codes too)
    let mut reposition_failures = 0;
    for seed in 0..N_SEEDS {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut world = World::from_layout(SAMPLE_LAYOUT, &WorldConfig::default());
        world.reset();
        let repositioned = world.reposition_entities(&mut rng).is_some();
        world.update_all_discovered();
        if !repositioned {
            reposition_failures += 1;
        }

        let player_id = world.ecs.player_id();
        let mut enemies: Vec<(usize, usize)> =
            world.ecs.enemy_ids().into_iter().map(|id| world.ecs.position_of(id)).collect();
        enemies.sort(); // the order of the ids is not part of the contract
        let goal = world.find_tiles_position_by_kind(TileKind::Goal);

        let observation = world.observation();
        let mine = observation.entities.iter().find(|e| e.id == player_id).expect("player observed");
        let cells: Vec<String> = mine.grid.iter().flatten().map(|c| c.to_string()).collect();

        writeln!(
            out,
            "reposition seed={seed} ok={repositioned} player={:?} enemies={enemies:?} goal={goal:?}\nobs {}",
            world.ecs.position_of(player_id),
            cells.join(",")
        )
        .unwrap();
    }
    writeln!(out, "SUMMARY reposition: {reposition_failures}/{N_SEEDS} returned None").unwrap();

    out
}

//-----------------------------------------------------
// Comparison
//-----------------------------------------------------

fn compare_with_golden(actual: &str) {
    if std::env::var("UPDATE_GOLDEN").is_ok() {
        fs::create_dir_all(GOLDEN_DIR).expect("cannot create the golden folder");
        fs::write(GOLDEN_PATH, actual).expect("cannot write the golden file");
        return;
    }

    let expected = fs::read_to_string(GOLDEN_PATH)
        .expect("golden file missing: run once with UPDATE_GOLDEN=1 on the unchanged engine");

    for (i, (e, a)) in expected.lines().zip(actual.lines()).enumerate() {
        if e != a {
            panic!("golden mismatch at line {}:\n  expected: {e}\n  actual:   {a}", i + 1);
        }
    }
    assert_eq!(expected.lines().count(), actual.lines().count(), "golden file and output differ in length");
}

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_outputs_are_deterministic_within_one_run() {
        // If this fails the golden file would be flaky: fix the source of randomness first
        assert_eq!(fingerprint(), fingerprint());
    }

    #[test]
    fn golden_outputs_match_the_saved_file() {
        let actual = fingerprint();
        for line in actual.lines().filter(|l| l.starts_with("SUMMARY")) {
            println!("{line}");
        }
        compare_with_golden(&actual);
    }
}
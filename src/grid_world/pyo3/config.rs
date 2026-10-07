//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use pyo3::prelude::*;
use crate::grid_world::world::state::WorldConfig;
use crate::grid_world::layout::generate::GenerationConfig;
use crate::grid_world::layout::mutate::MutationConfig;

//-----------------------------------------------------
// World Config Getters / Setters
//-----------------------------------------------------

#[pyclass(name = "WorldConfig", from_py_object)]
#[derive(Clone)]
pub struct PyWorldConfig {
    #[pyo3(get, set)]
    pub max_tick: u32,
    #[pyo3(get, set)]
    pub player_perception_range: usize,
    #[pyo3(get, set)]
    pub enemy_perception_range: usize,
    #[pyo3(get, set)]
    pub min_player_goal_distance: usize,
}

//-----------------------------------------------------
// World Config Constructor
//-----------------------------------------------------

#[pymethods]
impl PyWorldConfig {
    #[new]
    #[pyo3(signature = (
        max_tick=100, 
        player_perception_range=5, 
        enemy_perception_range=5, 
        min_player_goal_distance=0
    ))]
    fn new(
        max_tick: u32, 
        player_perception_range: usize, 
        enemy_perception_range: usize, 
        min_player_goal_distance: usize
    ) -> Self {
        PyWorldConfig { 
            max_tick, 
            player_perception_range, 
            enemy_perception_range, 
            min_player_goal_distance }
    }
}

//-----------------------------------------------------
// World Config Python <-> Rust
//-----------------------------------------------------

impl From<PyWorldConfig> for WorldConfig {
    fn from(cfg: PyWorldConfig) -> Self {
        WorldConfig {
            max_tick: cfg.max_tick,
            player_perception_range: cfg.player_perception_range,
            enemy_perception_range: cfg.enemy_perception_range,
            min_player_goal_distance: cfg.min_player_goal_distance,
        }
    }
}

//-----------------------------------------------------
// World Config Rust <-> Python
//-----------------------------------------------------

impl From<WorldConfig> for PyWorldConfig {
    fn from(cfg: WorldConfig) -> Self {
        PyWorldConfig {
            max_tick: cfg.max_tick,
            player_perception_range: cfg.player_perception_range,
            enemy_perception_range: cfg.enemy_perception_range,
            min_player_goal_distance: cfg.min_player_goal_distance,
        }
    }
}

//-----------------------------------------------------
// Generation Config Getters / Setters
//-----------------------------------------------------

#[pyclass(name = "GenerationConfig", from_py_object)]
#[derive(Clone)]
pub struct PyGenerationConfig {
    #[pyo3(get, set)]
    pub width: usize,
    #[pyo3(get, set)]
    pub height: usize,
    #[pyo3(get, set)]
    pub wall_density: f64,
    #[pyo3(get, set)]
    pub num_enemies: usize,
    #[pyo3(get, set)]
    pub num_traps: usize,
    #[pyo3(get, set)]
    pub min_player_goal_distance: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

//-----------------------------------------------------
// Generation Config Constructor
//-----------------------------------------------------

#[pymethods]
impl PyGenerationConfig {
    #[new]
    #[pyo3(signature = (
        width=16,
        height=16,
        wall_density=0.2,
        num_enemies=1,
        num_traps=0,
        min_player_goal_distance=0,
        seed=Some(0)
    ))]
    fn new(
        width: usize,
        height: usize,
        wall_density: f64,
        num_enemies: usize,
        num_traps: usize,
        min_player_goal_distance: usize,
        seed: Option<u64>,
    ) -> Self {
        PyGenerationConfig {
            width, height, wall_density,
            num_enemies, num_traps,
            min_player_goal_distance,
            seed,
        }
    }
}

//-----------------------------------------------------
// Generation Config Python <-> Rust
//-----------------------------------------------------

impl From<PyGenerationConfig> for GenerationConfig {
    fn from(cfg: PyGenerationConfig) -> Self {
        GenerationConfig {
            width: cfg.width,
            height: cfg.height,
            wall_density: cfg.wall_density,
            num_enemies: cfg.num_enemies,
            num_traps: cfg.num_traps,
            min_player_goal_distance: cfg.min_player_goal_distance,
            seed: cfg.seed,
        }
    }
}

impl From<GenerationConfig> for PyGenerationConfig {
    fn from(cfg: GenerationConfig) -> Self {
        PyGenerationConfig {
            width: cfg.width,
            height: cfg.height,
            wall_density: cfg.wall_density,
            num_enemies: cfg.num_enemies,
            num_traps: cfg.num_traps,
            min_player_goal_distance: cfg.min_player_goal_distance,
            seed: cfg.seed,
        }
    }
}

//-----------------------------------------------------
// Mutation Config Getters / Setters
//-----------------------------------------------------
#[pyclass(name = "MutationConfig", from_py_object)]
#[derive(Clone)]
pub struct PyMutationConfig {
    #[pyo3(get, set)]
    pub n_wall_mutations: usize,
    #[pyo3(get, set)]
    pub max_wall_density: f64,

    #[pyo3(get, set)]
    pub max_enemies: usize,
    #[pyo3(get, set)]
    pub enemy_add_weight: f64,
    #[pyo3(get, set)]
    pub enemy_remove_weight: f64,
    #[pyo3(get, set)]
    pub enemy_nothing_weight: f64,

    #[pyo3(get, set)]
    pub max_traps: usize,
    #[pyo3(get, set)]
    pub trap_add_weight: f64,
    #[pyo3(get, set)]
    pub trap_remove_weight: f64,
    #[pyo3(get, set)]
    pub trap_nothing_weight: f64,

    #[pyo3(get, set)]
    pub n_repositions: usize,
    #[pyo3(get, set)]
    pub min_player_goal_distance: usize,

    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

//-----------------------------------------------------
// Mutation Config Constructor
//-----------------------------------------------------

#[pymethods]
impl PyMutationConfig {
    #[new]
    #[pyo3(signature = (
        n_wall_mutations=3,
        max_wall_density=0.3,
        max_enemies=3,
        enemy_add_weight=0.4,
        enemy_remove_weight=0.2,
        enemy_nothing_weight=0.4,
        max_traps=3,
        trap_add_weight=0.2,
        trap_remove_weight=0.1,
        trap_nothing_weight=0.7,
        n_repositions=1,
        min_player_goal_distance=0,
        seed=Some(0)
    ))]
    fn new(
        n_wall_mutations: usize,
        max_wall_density: f64,
        max_enemies: usize,
        enemy_add_weight: f64,
        enemy_remove_weight: f64,
        enemy_nothing_weight: f64,
        max_traps: usize,
        trap_add_weight: f64,
        trap_remove_weight: f64,
        trap_nothing_weight: f64,
        n_repositions: usize,
        min_player_goal_distance: usize,
        seed: Option<u64>,
    ) -> Self {
        PyMutationConfig {
            n_wall_mutations, max_wall_density,
            max_enemies, enemy_add_weight, enemy_remove_weight, enemy_nothing_weight,
            max_traps, trap_add_weight, trap_remove_weight, trap_nothing_weight,
            n_repositions, min_player_goal_distance,
            seed,
        }
    }
}

//-----------------------------------------------------
// Mutation Config Python <-> Rust
//-----------------------------------------------------

impl From<PyMutationConfig> for MutationConfig {
    fn from(cfg: PyMutationConfig) -> Self {
        MutationConfig {
            n_wall_mutations: cfg.n_wall_mutations,
            max_wall_density: cfg.max_wall_density,
            max_enemies: cfg.max_enemies,
            enemy_add_weight: cfg.enemy_add_weight,
            enemy_remove_weight: cfg.enemy_remove_weight,
            enemy_nothing_weight: cfg.enemy_nothing_weight,
            max_traps: cfg.max_traps,
            trap_add_weight: cfg.trap_add_weight,
            trap_remove_weight: cfg.trap_remove_weight,
            trap_nothing_weight: cfg.trap_nothing_weight,
            n_repositions: cfg.n_repositions,
            min_player_goal_distance: cfg.min_player_goal_distance,
            seed: cfg.seed,
        }
    }
}

impl From<MutationConfig> for PyMutationConfig {
    fn from(cfg: MutationConfig) -> Self {
        PyMutationConfig {
            n_wall_mutations: cfg.n_wall_mutations,
            max_wall_density: cfg.max_wall_density,
            max_enemies: cfg.max_enemies,
            enemy_add_weight: cfg.enemy_add_weight,
            enemy_remove_weight: cfg.enemy_remove_weight,
            enemy_nothing_weight: cfg.enemy_nothing_weight,
            max_traps: cfg.max_traps,
            trap_add_weight: cfg.trap_add_weight,
            trap_remove_weight: cfg.trap_remove_weight,
            trap_nothing_weight: cfg.trap_nothing_weight,
            n_repositions: cfg.n_repositions,
            min_player_goal_distance: cfg.min_player_goal_distance,
            seed: cfg.seed,
        }
    }
}
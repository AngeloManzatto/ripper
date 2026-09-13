//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use pyo3::prelude::*;
use std::collections::HashMap;

use crate::grid_world::entity::EntityKind;
use crate::grid_world::world::{self, Observation, EndReason, StepResult, get_entity_ids_by_kind};
use crate::grid_world::world::World;
use crate::grid_world::environment::Environment;
use crate::grid_world::action::Action;
use crate::grid_world::layout;

//-----------------------------------------------------
// Layout Generate Config
//-----------------------------------------------------

#[pyclass(name = "GenerationConfig")]
pub struct PyGenerationConfig {
     #[pyo3(get)]
    pub width: usize,
     #[pyo3(get)]
    pub height: usize,
     #[pyo3(get)]
    pub wall_density: f64,
     #[pyo3(get)]
    pub num_enemies: usize,
     #[pyo3(get)]
    pub num_traps: usize,
}

#[pymethods]
impl PyGenerationConfig {
    #[new]
    #[pyo3(signature = (
        width=None,
        height=None,
        wall_density=None,
        num_enemies=None,
        num_traps=None,
    ))]
    fn new(
        width: Option<usize>,
        height: Option<usize>,
        wall_density: Option<f64>,
        num_enemies: Option<usize>,
        num_traps: Option<usize>,
    ) -> Self {
        let defaults = layout::GenerationConfig::default();
        PyGenerationConfig {
            width: width.unwrap_or(defaults.width),
            height: height.unwrap_or(defaults.height),
            wall_density: wall_density.unwrap_or(defaults.wall_density),
            num_enemies: num_enemies.unwrap_or(defaults.num_enemies),
            num_traps: num_traps.unwrap_or(defaults.num_traps),
        }
    }
}

impl From<&PyGenerationConfig> for layout::GenerationConfig {
    fn from(config: &PyGenerationConfig) -> Self {
        layout::GenerationConfig {
            width: config.width,
            height: config.height,
            wall_density: config.wall_density,
            num_enemies: config.num_enemies,
            num_traps: config.num_traps,
        }
    }
}

//-----------------------------------------------------
// Layout Mutation Config
//-----------------------------------------------------

#[pyclass(name = "MutationConfig")]
pub struct PyMutationConfig {
    #[pyo3(get)]
    pub n_wall_mutations: usize,
    #[pyo3(get)]
    pub max_wall_density: f64,

    #[pyo3(get)]
    pub max_enemies: u32,
    #[pyo3(get)]
    pub enemy_add_weight: f64,
    #[pyo3(get)]
    pub enemy_remove_weight: f64,
    #[pyo3(get)]
    pub enemy_nothing_weight: f64,

    #[pyo3(get)]
    pub max_traps: u32,
    #[pyo3(get)]
    pub trap_add_weight: f64,
    #[pyo3(get)]
    pub trap_remove_weight: f64,
    #[pyo3(get)]
    pub trap_nothing_weight: f64,

    #[pyo3(get)]
    pub n_repositions: usize,
}

#[pymethods]
impl PyMutationConfig {
    #[new]
    #[pyo3(signature = (
        n_wall_mutations=None,
        max_wall_density=None,
        max_enemies=None,
        enemy_add_weight=None,
        enemy_remove_weight=None,
        enemy_nothing_weight=None,
        max_traps=None,
        trap_add_weight=None,
        trap_remove_weight=None,
        trap_nothing_weight=None,
        n_repositions=None,
    ))]
    fn new(
        n_wall_mutations: Option<usize>,
        max_wall_density: Option<f64>,
        max_enemies: Option<u32>,
        enemy_add_weight: Option<f64>,
        enemy_remove_weight: Option<f64>,
        enemy_nothing_weight: Option<f64>,
        max_traps: Option<u32>,
        trap_add_weight: Option<f64>,
        trap_remove_weight: Option<f64>,
        trap_nothing_weight: Option<f64>,
        n_repositions: Option<usize>,
    ) -> Self {
        let defaults = layout::MutationConfig::default();
        PyMutationConfig {
            n_wall_mutations: n_wall_mutations.unwrap_or(defaults.n_wall_mutations),
            max_wall_density: max_wall_density.unwrap_or(defaults.max_wall_density),
            max_enemies: max_enemies.unwrap_or(defaults.max_enemies),
            enemy_add_weight: enemy_add_weight.unwrap_or(defaults.enemy_add_weight),
            enemy_remove_weight: enemy_remove_weight.unwrap_or(defaults.enemy_remove_weight),
            enemy_nothing_weight: enemy_nothing_weight.unwrap_or(defaults.enemy_nothing_weight),
            max_traps: max_traps.unwrap_or(defaults.max_traps),
            trap_add_weight: trap_add_weight.unwrap_or(defaults.trap_add_weight),
            trap_remove_weight: trap_remove_weight.unwrap_or(defaults.trap_remove_weight),
            trap_nothing_weight: trap_nothing_weight.unwrap_or(defaults.trap_nothing_weight),
            n_repositions: n_repositions.unwrap_or(defaults.n_repositions),
        }
    }
}

impl From<&PyMutationConfig> for layout::MutationConfig {
    fn from(config: &PyMutationConfig) -> Self {
        layout::MutationConfig {
            n_wall_mutations: config.n_wall_mutations,
            max_wall_density: config.max_wall_density,
            max_enemies: config.max_enemies,
            enemy_add_weight: config.enemy_add_weight,
            enemy_remove_weight: config.enemy_remove_weight,
            enemy_nothing_weight: config.enemy_nothing_weight,
            max_traps: config.max_traps,
            trap_add_weight: config.trap_add_weight,
            trap_remove_weight: config.trap_remove_weight,
            trap_nothing_weight: config.trap_nothing_weight,
            n_repositions: config.n_repositions,
        }
    }
}

//-----------------------------------------------------
// Generate Valid Layout
//-----------------------------------------------------

#[pyfunction]
pub fn generate_valid_layout_py(config: &PyGenerationConfig, max_attempts: u32) -> String {
    let rust_config: layout::GenerationConfig = config.into();
    layout::generate_valid_layout(&rust_config, max_attempts)
}

//-----------------------------------------------------
// Mutate Valid Layout
//-----------------------------------------------------

#[pyfunction]
pub fn mutate_valid_layout_py(layout_str: String, config: &PyMutationConfig, max_attempts: u32) -> String {
    let rust_config: layout::MutationConfig = config.into();
    layout::mutate_valid_layout(&layout_str, &rust_config, max_attempts)
}

//-----------------------------------------------------
// Step Result
//-----------------------------------------------------

#[pyclass(name = "EndReason", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyEndReason {
    GoalReached,
    Caught,
    Trapped,
    Timeout,
}

impl From<EndReason> for PyEndReason {
    fn from(reason: EndReason) -> Self {
        match reason {
            EndReason::GoalReached => PyEndReason::GoalReached,
            EndReason::Caught      => PyEndReason::Caught,
            EndReason::Trapped     => PyEndReason::Trapped,
            EndReason::Timeout     => PyEndReason::Timeout,
        }
    }
}

//-----------------------------------------------------
// Action
//-----------------------------------------------------

#[pyclass(name = "Action", from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyAction {
    Up,
    Down,
    Left,
    Right,
}

impl From<PyAction> for Action {
    fn from(action: PyAction) -> Self {
        match action {
            PyAction::Up    => Action::Up,
            PyAction::Down  => Action::Down,
            PyAction::Left  => Action::Left,
            PyAction::Right => Action::Right,
        }
    }
}

//-----------------------------------------------------
// Observation
//-----------------------------------------------------

#[pyclass(name = "Observation")]
pub struct PyObservation {
    #[pyo3(get)]
    pub grid: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub player_id: u32,
    #[pyo3(get)]
    pub enemy_ids: Vec<u32>,
}

impl From<Observation> for PyObservation {
    fn from(obs: Observation) -> Self {
        PyObservation {
            grid: obs.grid,
            player_id: obs.player_id,
            enemy_ids: obs.enemy_ids,
        }
    }
}

//-----------------------------------------------------
// Step Result
//-----------------------------------------------------
#[pyclass(name = "StepResult")]
pub struct PyStepResult {
    #[pyo3(get)]
    pub done: bool,
    #[pyo3(get)]
    pub reason: Option<PyEndReason>,
}

impl From<StepResult> for PyStepResult {
    fn from(result: StepResult) -> Self {
        let reason = result.reason.map(|r| r.into());
        PyStepResult { done: result.done, reason }
    }
}

//-----------------------------------------------------
// World
//-----------------------------------------------------
#[pyclass]
pub struct PyWorld {
    world: World,
}

#[pymethods]
impl PyWorld {
    #[new]
    fn new(layout: &str, max_tick: u32) -> Self {
        let world = layout::parse_layout(layout, max_tick);
        PyWorld { world }
    }

    fn get_grid_shape(&self) ->(usize, usize) {
        (self.world.width, self.world.height)
    }

    fn get_player_id(&self) -> u32 {
        *get_entity_ids_by_kind(&self.world, EntityKind::Player)
            .first()
            .expect("no player entity found")
    }

    fn get_enemies_ids(&self) -> Vec<u32> {
        get_entity_ids_by_kind(&self.world, EntityKind::Enemy)
    }

    fn reset(&mut self) {
        self.world.reset();
    }

    fn step(&mut self, actions: HashMap<u32, PyAction>) -> PyResult<PyStepResult> {
        let parsed: HashMap<u32, Action> = actions.into_iter()
            .map(|(id, py_action)| (id, py_action.into()))
            .collect();

        let result = self.world.step(parsed);
        Ok(result.into())
    }

    fn observation(&self) -> PyObservation {
        self.world.observation().into()
    }

    fn print_world(&self) {
        world::print_world(&self.world);
    }

    fn __repr__(&self) -> String {
        let player_id = get_entity_ids_by_kind(&self.world, EntityKind::Player).first().copied();
        format!("<PyWorld width={} height={} tick={} done={} player_id={:?}>",
            self.world.width, self.world.height, self.world.tick, self.world.done, player_id)
    }

    fn __str__(&self) -> String {
        world::render_world(&self.world)
    }
}
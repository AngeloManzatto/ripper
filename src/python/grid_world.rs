//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use pyo3::prelude::*;
use std::collections::HashMap;

use crate::grid_world::world::{Observation, StepResult};
use crate::grid_world::world::World;
use crate::grid_world::environment::Environment;
use crate::grid_world::action::Action;
use crate::grid_world::layout::parse_layout;

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
    pub reason: Option<String>, 
}

impl From<StepResult> for PyStepResult {
    fn from(result: StepResult) -> Self {
        let reason = result.reason.map(|r| format!("{:?}", r)); // uses EndReason's Debug derive
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
        let world = parse_layout(layout, max_tick);
        PyWorld { world }
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
}
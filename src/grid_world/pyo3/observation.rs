//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use pyo3::prelude::*;
use crate::grid_world::entity::EntityStatus;
use crate::grid_world::world::observation::{TimeStep, Observation, EntityObservation};

//-----------------------------------------------------
// EntityStatus -> &str
//-----------------------------------------------------

fn entity_status_to_str(status: EntityStatus) -> &'static str {
    match status {
        EntityStatus::Alive => "Alive",
        EntityStatus::GoalReached => "GoalReached",
        EntityStatus::Trapped => "Trapped",
        EntityStatus::Caught => "Caught",
    }
}

//-----------------------------------------------------
// Entity Observation
//-----------------------------------------------------

#[pyclass(name = "EntityObservation", from_py_object)]
#[derive(Clone)]
pub struct PyEntityObservation {
    #[pyo3(get)]
    pub id: u32,
    #[pyo3(get)]
    pub position: (usize, usize),
    #[pyo3(get)]
    pub grid: Vec<Vec<i32>>,
    #[pyo3(get)]
    pub status: &'static str,
}

impl From<EntityObservation> for PyEntityObservation {
    fn from(eo: EntityObservation) -> Self {
        PyEntityObservation {
            id: eo.id,
            position: eo.position,
            grid: eo.grid,
            status: entity_status_to_str(eo.status),
        }
    }
}

//-----------------------------------------------------
// Observation
//-----------------------------------------------------

#[pyclass(name = "Observation", from_py_object)]
#[derive(Clone)]
pub struct PyObservation {
    #[pyo3(get)]
    pub player_id: u32,
    #[pyo3(get)]
    pub enemy_ids: Vec<u32>,
    #[pyo3(get)]
    pub entities: Vec<PyEntityObservation>,
}

impl From<Observation> for PyObservation {
    fn from(obs: Observation) -> Self {
        PyObservation {
            player_id: obs.player_id,
            enemy_ids: obs.enemy_ids,
            entities: obs.entities.into_iter().map(Into::into).collect(),
        }
    }
}

//-----------------------------------------------------
// TimeStep
//-----------------------------------------------------

#[pyclass(name = "TimeStep")]
pub struct PyTimeStep {
    #[pyo3(get)]
    pub observation: PyObservation,
    #[pyo3(get)]
    pub terminated: bool,
    #[pyo3(get)]
    pub truncated: bool,
}

#[pymethods]
impl PyTimeStep {
    fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

impl From<TimeStep> for PyTimeStep {
    fn from(ts: TimeStep) -> Self {
        PyTimeStep {
            observation: ts.observation.into(),
            terminated: ts.terminated,
            truncated: ts.truncated,
        }
    }
}
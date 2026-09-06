//-----------------------------------------------------
// Imports
//-----------------------------------------------------
use std::collections::HashMap;
use crate::grid_world::world::StepResult;

//-----------------------------------------------------
// Environment
//-----------------------------------------------------
pub trait Environment {
    type Action;
    type Observation;

    fn reset(&mut self);
    fn step(&mut self, actions: HashMap<u32, Self::Action>) -> StepResult;
    fn observation(&self) -> Self::Observation;
    fn is_done(&self) -> bool;
}
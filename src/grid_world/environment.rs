//-----------------------------------------------------
// Imports
//-----------------------------------------------------
use std::collections::HashMap;
use crate::grid_world::world::{TimeStep};

//-----------------------------------------------------
// Environment
//-----------------------------------------------------
pub trait Environment {
    type Action;
    type Observation;

    fn reset(&mut self) -> TimeStep;
    fn step(&mut self, actions: HashMap<u32, Self::Action>) -> TimeStep;
    fn observation(&self) -> Self::Observation;
    fn is_done(&self) -> bool;
}
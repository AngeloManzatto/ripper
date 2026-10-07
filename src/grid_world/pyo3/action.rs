//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use pyo3::prelude::*;
use crate::grid_world::action::Action;

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

//-----------------------------------------------------
// Action Python <-> Rust
//-----------------------------------------------------

impl From<PyAction> for Action {
    fn from(action: PyAction) -> Self {
        match action {
            PyAction::Up => Action::Up,
            PyAction::Down => Action::Down,
            PyAction::Left => Action::Left,
            PyAction::Right => Action::Right,
        }
    }
}

impl From<Action> for PyAction {
    fn from(action: Action) -> Self {
        match action {
            Action::Up => PyAction::Up,
            Action::Down => PyAction::Down,
            Action::Left => PyAction::Left,
            Action::Right => PyAction::Right,
        }
    }
}
//-----------------------------------------------------
// Modules
//-----------------------------------------------------

pub mod grid_world;

//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use pyo3::prelude::*;

use grid_world::pyo3::world::PyWorld;
use grid_world::pyo3::config::{PyWorldConfig, PyGenerationConfig, PyMutationConfig};
use grid_world::pyo3::action::PyAction;
use grid_world::pyo3::observation::{PyObservation, PyEntityObservation, PyTimeStep};
use grid_world::pyo3::layout::{generate_valid_layout_py, mutate_valid_layout_py};

//-----------------------------------------------------
// Grid World Bindings
//-----------------------------------------------------

#[pymodule]
fn ripper(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorld>()?;
    m.add_class::<PyWorldConfig>()?;
    m.add_class::<PyGenerationConfig>()?;
    m.add_class::<PyMutationConfig>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyObservation>()?;
    m.add_class::<PyEntityObservation>()?;
    m.add_class::<PyTimeStep>()?;

    m.add_function(wrap_pyfunction!(generate_valid_layout_py, m)?)?;
    m.add_function(wrap_pyfunction!(mutate_valid_layout_py, m)?)?;

    Ok(())
}
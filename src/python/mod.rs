//-----------------------------------------------------
// Modules
//-----------------------------------------------------

pub mod grid_world;  

//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use pyo3::prelude::*;

//-----------------------------------------------------
// Grid World Bindings
//-----------------------------------------------------

#[pymodule]
fn ripper(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<grid_world::PyWorldConfig>()?;
    m.add_class::<grid_world::PyWorld>()?;
    m.add_class::<grid_world::PyObservation>()?;
    m.add_class::<grid_world::PyEntityObservation>()?;
    m.add_class::<grid_world::PyStepResult>()?;
    m.add_class::<grid_world::PyEndReason>()?;
    m.add_class::<grid_world::PyAction>()?;
    m.add_class::<grid_world::PyGenerationConfig>()?;
    m.add_class::<grid_world::PyMutationConfig>()?;
    m.add_function(wrap_pyfunction!(grid_world::generate_valid_layout_py, m)?)?;
    m.add_function(wrap_pyfunction!(grid_world::mutate_valid_layout_py, m)?)?;
    Ok(())
}
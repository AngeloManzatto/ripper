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
    m.add_class::<grid_world::PyWorld>()?;
    m.add_class::<grid_world::PyObservation>()?;
    m.add_class::<grid_world::PyStepResult>()?;
    Ok(())
}
//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use std::collections::HashMap;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::grid_world::world::state::World;
use crate::grid_world::pyo3::config::PyWorldConfig;
use crate::grid_world::pyo3::observation::PyTimeStep;
use crate::grid_world::pyo3::action::PyAction;

//-----------------------------------------------------
// PyWorld
//-----------------------------------------------------

#[pyclass(name = "World")]
pub struct PyWorld {
    world: World,
    rng: StdRng,
}

//-----------------------------------------------------
// PyWorld Constructor
//-----------------------------------------------------

#[pymethods]
impl PyWorld {
    #[new]
    #[pyo3(signature = (layout, config=None, seed=None))]
    fn new(layout: String, config: Option<PyWorldConfig>, seed: Option<u64>) -> Self {
        let config = config.map(Into::into).unwrap_or_default();
        let world = World::from_layout(&layout, &config);

        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        PyWorld { world, rng }
    }

    //-------------------------------------------------
    // Lifecycle
    //-------------------------------------------------

    #[pyo3(signature = (reposition=false))]
    fn reset(&mut self, reposition: bool) -> PyTimeStep {
        if reposition {
            self.world.reset_with_reposition(&mut self.rng).into()
        } else {
            self.world.reset().into()
        }
    }

    fn step(&mut self, actions: HashMap<u32, PyAction>) -> PyTimeStep {
        let actions = actions.into_iter().map(|(id, a)| (id, a.into())).collect();
        self.world.step(actions, &mut self.rng).into()
    }

    //-------------------------------------------------
    // Rendering
    //-------------------------------------------------

    fn render_world(&self) -> String {
        self.world.render_world()
    }

    fn render_entity_view(&self, id: u32) -> String {
        self.world.render_entity_view(id)
    }

    //-------------------------------------------------
    // Read-only queries
    //-------------------------------------------------

    #[getter]
    fn width(&self) -> usize {
        self.world.width
    }

    #[getter]
    fn height(&self) -> usize {
        self.world.height
    }

    #[getter]
    fn tick(&self) -> u32 {
        self.world.tick
    }

    #[getter]
    fn done(&self) -> bool {
        self.world.done
    }
}
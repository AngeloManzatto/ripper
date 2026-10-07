//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::grid_world::layout::generate::{self, GenerationConfig};
use crate::grid_world::layout::mutate::{self, MutationConfig};
use crate::grid_world::pyo3::config::{PyGenerationConfig, PyMutationConfig};

//-----------------------------------------------------
// Generate a valid layout
//-----------------------------------------------------

#[pyfunction]
#[pyo3(name = "generate_valid_layout")]
pub fn generate_valid_layout_py(config: PyGenerationConfig) -> Option<String> {
    let seed = config.seed;
    let config: GenerationConfig = config.into();

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    generate::generate_valid_layout(&config, &mut rng)
}

//-----------------------------------------------------
// Mutate into a valid layout
//-----------------------------------------------------

#[pyfunction]
#[pyo3(name = "mutate_valid_layout")]
pub fn mutate_valid_layout_py(layout: String, config: PyMutationConfig) -> Option<String> {
    let seed = config.seed;
    let config: MutationConfig = config.into();

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    mutate::mutate_valid_layout(&layout, &config, &mut rng)
}
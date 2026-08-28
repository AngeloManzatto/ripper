pub mod grid;

use pyo3::prelude::*;

#[pyclass]
struct PyGrid {
    grid: grid::Grid,
    player_id: u32, // convenience — same id grid.player_id already tracks
}

#[pymethods]
impl PyGrid {
    #[new]
    fn new() -> Self {
        let config = grid::GridConfig {
            agent_start: (0, 0),
            goal_pos: (9, 9),
            enemy_start: (5, 5),
        };
        let g = grid::create_grid(10, 10, config);
        let player_id = g.player_id.expect("no player id");
        PyGrid { grid: g, player_id }
    }

    fn reset(&mut self) {
        grid::reset_grid(&mut self.grid);
    }
}

#[pymodule]
fn ripper(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGrid>()?;
    Ok(())
}
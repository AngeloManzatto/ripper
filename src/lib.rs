pub mod grid;

use pyo3::prelude::*;

#[pyclass]
struct PyGrid {
    grid: grid::Grid,
    player_id: u32,
}

#[pymethods]
impl PyGrid {

    // Initialize grid
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

    // Step
    fn step(&mut self, action: &str) -> PyResult<(f64, bool)> {
        let parsed_action = match action {
            "up"    => grid::Action::Up,
            "down"  => grid::Action::Down,
            "left"  => grid::Action::Left,
            "right" => grid::Action::Right,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("invalid action")),
        };

        let result = grid::step(&mut self.grid, self.player_id, parsed_action);
        Ok((result.reward, result.done))
    }

    // Reset grid
    fn reset(&mut self) {
        grid::reset_grid(&mut self.grid);
    }

    // Get current observation
    fn observation(&self) -> Vec<Vec<Vec<f64>>> {
        grid::get_observation(&self.grid)
    }
}

#[pymodule]
fn ripper(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGrid>()?;
    Ok(())
}
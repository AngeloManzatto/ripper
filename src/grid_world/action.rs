//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use rand::{
    Rng,
    distributions::{Distribution, Standard},
};

//-----------------------------------------------------
// Action
//-----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

//-----------------------------------------------------
// Action to delta
//-----------------------------------------------------

impl Action {
    pub fn to_delta(&self) -> (i32, i32) {
        match self {
            Action::Up => (-1, 0),
            Action::Down => (1, 0),
            Action::Left => (0, -1),
            Action::Right => (0, 1),
        }
    }
}

impl Distribution<Action> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        match rng.gen_range(0..=3) {
            // rand 0.8
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            _ => Action::Right,
        }
    }
}


//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use rand::{
    distributions::{Distribution, Standard},
    Rng,
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

impl Distribution<Action> for Standard  {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        match rng.gen_range(0..=3) { // rand 0.8
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            _ => Action::Right, 
        }
    }
}

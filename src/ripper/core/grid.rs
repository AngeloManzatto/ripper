//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

//-----------------------------------------------------
// Grid
//-----------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid<T> {
    width: usize,
    height: usize,
    cells: Vec<T>,
}

//-----------------------------------------------------
// Grid constructor
//-----------------------------------------------------

impl<T: Clone> Grid<T> {
    pub fn new(width: usize, height: usize, fill: T) -> Grid<T> {
        Grid {
            width,
            height,
            cells: vec![fill; width * height],
        }
    }
}

//-----------------------------------------------------
// Grid methods
//-----------------------------------------------------

impl<T> Grid<T> {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn in_bounds(&self, pos: (usize, usize)) -> bool {
        pos.0 < self.height && pos.1 < self.width
    }

    pub fn get(&self, pos: (usize, usize)) -> Option<&T> {
        if !self.in_bounds(pos) {
            return None;
        }
        Some(&self.cells[pos.0 * self.width + pos.1])
    }

    pub fn get_mut(&mut self, pos: (usize, usize)) -> Option<&mut T> {
        if !self.in_bounds(pos) {
            return None;
        }
        let index = pos.0 * self.width + pos.1;
        Some(&mut self.cells[index])
    }
    
}   
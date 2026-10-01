//-----------------------------------------------------
// Manhattan Distance
//-----------------------------------------------------

pub fn manhattan_distance(a: (usize, usize), b: (usize, usize)) -> i32 {
    (a.0 as i32 - b.0 as i32).abs() + (a.1 as i32 - b.1 as i32).abs()
}

//-----------------------------------------------------
// Neighbors
//-----------------------------------------------------

pub fn neighbors(pos: (usize, usize)) -> Vec<(i32, i32)> {

    // Creates a vector with all directions delta

    let (x, y) = (pos.0 as i32, pos.1 as i32);
    vec![(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
}

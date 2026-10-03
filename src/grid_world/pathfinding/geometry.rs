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

//-----------------------------------------------------
// Trace Line
//-----------------------------------------------------

pub fn trace_line(from: (usize, usize), to: (usize, usize)) -> Vec<(usize, usize)> {
    let mut points = Vec::new();

    let mut x0 = from.0 as i32;
    let mut y0 = from.1 as i32;
    let x1 = to.0 as i32;
    let y1 = to.1 as i32;

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs(); // note the negative sign, explained below
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        points.push((x0 as usize, y0 as usize));

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }

    points
}

//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use crate::grid_world::world;

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
    let dy = -(y1 - y0).abs();  // note the negative sign, explained below
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

//-----------------------------------------------------
// Has Line of Sight (LOS)
//-----------------------------------------------------

pub fn has_line_of_sight(
    world: &world::World, 
    from: (usize, usize), 
    to: (usize, usize), 
    perception_range: usize) -> bool 
{
    let dr = from.0 as i32 - to.0 as i32;
    let dc = from.1 as i32 - to.1 as i32;
    if (dr * dr + dc * dc) as usize > perception_range * perception_range {
        return false;
    }

    let mut x0 = from.0 as i32;
    let mut y0 = from.1 as i32;
    let x1 = to.0 as i32;
    let y1 = to.1 as i32;

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if (x0, y0) == (x1, y1) {
            return true;
        }

        let e2 = 2 * err;
        let step_x = e2 >= dy;
        let step_y = e2 <= dx;

        let prev = (x0 as usize, y0 as usize);

        if step_x { err += dy; x0 += sx; }
        if step_y { err += dx; y0 += sy; }

        let current = (x0 as usize, y0 as usize);

        if step_x && step_y {
            if is_diagonal_corner_blocked(world, prev, current) {
                return false;
            }
        }

        if current == (x1 as usize, y1 as usize) {
            return true;  // reached the target — visible, whether it's a wall or floor
        }

        if !world::is_walkable(world, current) {
            return false;  // an intermediate (non-target) wall still blocks the path
        }

        
    }
}

//-----------------------------------------------------
// Check if flanks are blocked
//-----------------------------------------------------

fn is_diagonal_corner_blocked(
    world: &world::World, 
    prev: (usize, usize), 
    current: (usize, usize)) -> bool 
{
    let flank_1 = (prev.0, current.1);
    let flank_2 = (current.0, prev.1);

    !world::is_walkable(world, flank_1) && !world::is_walkable(world, flank_2)
}

//-----------------------------------------------------
// Visible Line of Sign
//-----------------------------------------------------

pub fn get_visible_cells(
    world: &world::World, 
    from: (usize, usize), 
    perception_range: usize
) -> Vec<(usize, usize)> 
{
    let mut visible = Vec::new();

    let range = perception_range as i32;

    // Trace a square around the starting point
    let row_start = (from.0 as i32 - range).max(0) as usize;
    let row_end = (from.0 as i32 + range).min(world.height as i32 - 1) as usize;
    let col_start = (from.1 as i32 - range).max(0) as usize;
    let col_end = (from.1 as i32 + range).min(world.width as i32 - 1) as usize;

    for row in row_start..=row_end {
        for col in col_start..=col_end {
            let target = (row, col);
            if has_line_of_sight(world, from, target, perception_range) {
                visible.push(target);
            }
        }
    }

    visible
}

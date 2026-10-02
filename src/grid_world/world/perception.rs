//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use crate::grid_world::world::state::World;

//-----------------------------------------------------
// Perception
//-----------------------------------------------------

impl World {

    //-----------------------------------------------------
    // Has line of sight
    //-----------------------------------------------------
    pub fn has_line_of_sight(&self, from: (usize, usize), to: (usize, usize), perception_range: usize) -> bool {
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
                if self.is_diagonal_corner_blocked(prev, current) {
                    return false;
                }
            }

            if current == (x1 as usize, y1 as usize) {
                return true;  // reached the target — visible, whether it's a wall or floor
            }

            if !self.is_walkable(current) {
                return false;  // an intermediate (non-target) wall still blocks the path
            }

            
        }
    }

    //-----------------------------------------------------
    // is diagonal corner blocked
    //-----------------------------------------------------

    fn is_diagonal_corner_blocked(&self, prev: (usize, usize), current: (usize, usize)) -> bool 
    {
        let flank_1 = (prev.0, current.1);
        let flank_2 = (current.0, prev.1);

        !self.is_walkable(flank_1) && !self.is_walkable(flank_2)
    }

    //-----------------------------------------------------
    // Get visible cells
    //-----------------------------------------------------

    pub fn get_visible_cells(&self, from: (usize, usize), perception_range: usize) -> Vec<(usize, usize)> 
    {
        let mut visible = Vec::new();

        let range = perception_range as i32;

        // Trace a square around the starting point
        let row_start = (from.0 as i32 - range).max(0) as usize;
        let row_end = (from.0 as i32 + range).min(self.height as i32 - 1) as usize;
        let col_start = (from.1 as i32 - range).max(0) as usize;
        let col_end = (from.1 as i32 + range).min(self.width as i32 - 1) as usize;

        for row in row_start..=row_end {
            for col in col_start..=col_end {
                let target = (row, col);
                if self.has_line_of_sight(from, target, perception_range) {
                    visible.push(target);
                }
            }
        }

        visible
    }

}

//-----------------------------------------------------
// Update discovered
//-----------------------------------------------------

impl World {
    pub fn update_discovered(&mut self, id: u32, pos: (usize, usize), perception_range: usize) 
    {
        // Get visible cells from the entity stand point of view
        let visible = self.get_visible_cells(pos, perception_range);

        // Update discovered cells for this entity on ECS
        for cell in visible {
            self.ecs.set_discovered(id, cell);
        }
    }
}
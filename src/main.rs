//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use ripper::grid_world::entity;
use ripper::grid_world::layout;
use ripper::grid_world::world;

//-----------------------------------------------------
// Main Function
//-----------------------------------------------------

fn main() {

    let max_tick = 100;
    let config = layout::GenerationConfig {
        width: 16,
        height: 16,
        wall_density:0.2,
        num_enemies:1,
        num_traps:1
    };

    let layout = layout::generate_layout(&config);

    let world = layout::parse_layout(&layout, max_tick);

    world::render_world(&world);

}

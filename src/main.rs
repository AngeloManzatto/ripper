//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use ripper::grid_world::layout;
use ripper::grid_world::world;
use ripper::grid_world::render;

//-----------------------------------------------------
// Main Function
//-----------------------------------------------------

fn main() {

    let generation_config = layout::GenerationConfig {
        width: 16,
        height: 16,
        wall_density:0.2,
        num_enemies:1,
        num_traps:1,
        min_player_goal_distance:0
    };
    

    let layout = layout::generate_layout(&generation_config);
    let world = layout::parse_layout(&layout, &world::WorldConfig::default());

    render::render_world(&world);

   
}


//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use macroquad::prelude::*;
use std::collections::HashMap;

use ripper::grid_world::world::{World, get_entity_ids_by_kind};
use ripper::grid_world::entity::EntityKind;
use ripper::grid_world::action::Action;
use ripper::grid_world::environment::Environment;
use ripper::grid_world::layout;
use ripper::grid_world::world;
use ripper::grid_world::vision;
use ripper::grid_world::render;

//-----------------------------------------------------
// Main loop
//-----------------------------------------------------

#[macroquad::main("RIPPER")]
async fn main() {

    let config = layout::GenerationConfig {
        width: 16,
        height: 16,
        wall_density:0.2,
        num_enemies:1,
        num_traps:1,
        min_player_goal_distance:5
    };

    let mut world = generate_world(config, &world::WorldConfig::default());
    let mut player_id = get_entity_ids_by_kind(&world, EntityKind::Player)[0];
    let perception_range = 5;

    loop {
        // Clear screen buffer
        clear_background(BLACK);

        if is_key_pressed(KeyCode::R) {
            world = generate_world(config, &world::WorldConfig::default());
            player_id = get_entity_ids_by_kind(&world, EntityKind::Player)[0];
        }
        else if is_key_pressed(KeyCode::Q) {break }

        // Poll for a key press this frame, map to an Action
        let action = if is_key_pressed(KeyCode::W) { Some(Action::Up) }
            else if is_key_pressed(KeyCode::S) { Some(Action::Down) }
            else if is_key_pressed(KeyCode::A) { Some(Action::Left) }
            else if is_key_pressed(KeyCode::D) { Some(Action::Right) }
            else { None };

        if let Some(action) = action {
            let mut actions = HashMap::new();
            actions.insert(player_id, action);
            world.step(actions);
        }

        // Get player positions
        let player_pos = world.positions[&player_id];
        
        // Get visible cells for this entity
        let visible_cells = vision::get_visible_cells(
            &world, 
            player_pos, 
            perception_range);

        let discovered = world.discovered.get(&player_id).cloned().unwrap_or_default();
        // render world
        draw_world(&world, &visible_cells, &discovered);

        // Wait for next frame
        next_frame().await;
    }
}

fn generate_world(
    generation_config: layout::GenerationConfig, 
    world_config: &world::WorldConfig
) -> world::World
{
    let layout = layout::generate_layout(&generation_config);
    let world   = layout::parse_layout(&layout, &world_config);
    render::print_world(&world);
    world

}

fn draw_world(world: &World, visible_cells: &Vec<(usize, usize)>, discovered: &std::collections::HashSet<(usize, usize)>) {
    let cell_size = (screen_width() / world.width as f32).min(screen_height() / world.height as f32);

    for row in 0..world.height {
        for col in 0..world.width {
            let cell = (row, col);

            // Pass 2: mobile entities, only if currently visible
            let is_currently_visible = visible_cells.contains(&cell);
            let id = world.positions.iter().find_map(
                |(key, &val)| if val == cell { Some(key) } else { None }
            );

            let color = if let Some(found_id) = id {
                let kind = world.kinds.get(found_id).expect("ID not found in kinds");

                match kind {
                    EntityKind::Player | EntityKind::Enemy => {
                        if is_currently_visible { Some(match kind {
                            EntityKind::Player => GREEN,
                            EntityKind::Enemy => RED,
                            _ => unreachable!(),
                        }) } else { None }
                    }
                    _ => {
                        // Pass 1: terrain, shown if ever discovered
                        if discovered.contains(&cell) {
                            Some(match kind {
                                EntityKind::Goal => BLUE,
                                EntityKind::Wall => GRAY,
                                EntityKind::Trap => YELLOW,
                                _ => unreachable!(),
                            })
                        } else { None }
                    }
                }
            } else {
                None // empty cell, nothing to draw either way
            };

            if let Some(color) = color {
                let x = col as f32 * cell_size;
                let y = row as f32 * cell_size;
                draw_rectangle(x, y, cell_size, cell_size, color);
            }
        }
    }
}
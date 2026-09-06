//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use macroquad::prelude::*;
use std::collections::HashMap;

use ripper::grid_world::world::{World, get_entity_ids_by_kind};
use ripper::grid_world::entity::EntityKind;
use ripper::grid_world::action::Action;
use ripper::grid_world::environment::Environment;
use ripper::grid_world::layout::parse_layout;

//-----------------------------------------------------
// Main loop
//-----------------------------------------------------

#[macroquad::main("RIPPER")]
async fn main() {
    let layout = "\
        ##########
        #P.......#
        #....#...#
        #....#...#
        #...##..G#
        #...T....#
        #......E.#
        ##########";

    let mut world = parse_layout(layout, 100);
    let player_id = get_entity_ids_by_kind(&world, EntityKind::Player)[0];
    
    loop {
        // Clear screen buffer
        clear_background(BLACK);

        if is_key_pressed(KeyCode::R) {world.reset() }
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

        // render world
        draw_world(&world);

        // Wait for next frame
        next_frame().await;
    }
}

fn draw_world(world: &World) {

    let cell_size = (screen_width() / world.width as f32).min(screen_height() / world.height as f32);

    for row in 0..world.height {
        for col in 0..world.width {
            let x = col as f32 * cell_size;
            let y = row as f32 * cell_size;

            // Search ID based on position match
            let id = world.positions.iter().find_map(
                |(key, &val)| if val == (row, col) { Some(key) } else { None }
            );

            // Get entity kind by id
            if let Some(found_id) = id {
                let kind = world.kinds.get(found_id).expect("ID not found in kinds");
                
                let color = match kind {
                    EntityKind::Player => GREEN,
                    EntityKind::Enemy  => RED,
                    EntityKind::Goal   => BLUE,
                    EntityKind::Wall   => GRAY,
                    EntityKind::Trap   => YELLOW,
                };

                draw_rectangle(
                    x, 
                    y, 
                    cell_size, 
                    cell_size, 
                    color);

            }

        }
    }
}
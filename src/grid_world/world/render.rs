//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use colored::*;

use crate::grid_world::entity::{EntityKind, TileKind};
use crate::grid_world::world::state::{World,WorldConfig};
use crate::grid_world::world::perception::PerceivedCell;

//-----------------------------------------------------
// Tile color
//-----------------------------------------------------

fn tile_color(kind: TileKind) -> ColoredString {
    let ch = kind.to_char();

    match kind {
        TileKind::Wall => ch.to_string().white(),
        TileKind::Trap => ch.to_string().red(),
        TileKind::Goal => ch.to_string().green(),
        TileKind::Free => ch.to_string().normal(),
    }
}

//-----------------------------------------------------
// Entity color
//-----------------------------------------------------

fn entity_color(kind: EntityKind) -> ColoredString {
    let ch = kind.to_char();

    match kind {
        EntityKind::Player => ch.to_string().bright_cyan().bold(),
        EntityKind::Enemy => ch.to_string().bright_yellow(),
    }
}

//-----------------------------------------------------
// Renderers
//-----------------------------------------------------

impl World {
    //-----------------------------------------------------
    // Renderer world
    //-----------------------------------------------------
    pub fn render_world(&self) -> String {
        // Position -> id, for every live entity
        let entity_at = self.ecs.entities_at();

        let mut grid = vec![vec!["".normal(); self.width]; self.height];

        for row in 0..self.height {
            for col in 0..self.width {
                grid[row][col] = if let Some(&entity_id) = entity_at.get(&(row, col)) {
                    entity_color(self.ecs.kind_of(entity_id))
                } else {
                    tile_color(self.grid[row][col])
                };
            }
        }

        grid.iter()
            .map(|row| row.iter().map(|cell| cell.to_string()).collect::<String>())
            .collect::<Vec<String>>()
            .join("\n")
    }

    //-----------------------------------------------------
    // Renderer entity view
    //-----------------------------------------------------

    pub fn render_entity_view(&self, id: u32) -> String {
        let perceived = self.perceived_grid(id);

        let grid: Vec<Vec<ColoredString>> = perceived.iter().map(|row| {
            row.iter().map(|cell| match cell {
                PerceivedCell::Visible { occupant: Some(kind), .. } => entity_color(*kind),
                PerceivedCell::Visible { terrain, occupant: None } => tile_color(*terrain),
                PerceivedCell::Remembered { terrain } => tile_color(*terrain).dimmed(),
                PerceivedCell::Unknown => "X".truecolor(40, 40, 40),
            }).collect()
        }).collect();

        grid.iter()
            .map(|row| row.iter().map(|c| c.to_string()).collect::<String>())
            .collect::<Vec<String>>()
            .join("\n")
    }

}

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn render_world_visual() {
    use crate::grid_world::world::state::WorldConfig;
    use colored::control::set_override;

    set_override(true); // force ANSI colors even though cargo test pipes stdout (not a real tty)

    let world = World::from_layout(
        "#######\n\
         #P..T.#\n\
         #.##..#\n\
         #..E..#\n\
         #....G#\n\
         #######",
        &WorldConfig::default(),
    );

    println!("{}", world.render_world());
}


#[test]
fn render_entity_view_visual() {
    use colored::control::set_override;
    use std::collections::HashMap;
    use crate::grid_world::action::Action;
    set_override(true);

    let mut config = WorldConfig::default();
    config.player_perception_range = 2; // small on purpose, to make visible vs remembered vs unknown obvious

    let mut world = World::from_layout(
        "###########\n\
         #P.......G#\n\
         ###########",
        &config,
    );

    let player_id = world.ecs.player_id();
    let mut rng = rand::thread_rng();

    println!("=== Before moving (perception range 2) ===");
    println!("{}", world.render_entity_view(player_id));

    // Walk the player right a few times — far enough that the starting
    // area drops out of current sight but stays in memory.
    for _ in 0..4 {
        let mut actions = HashMap::new();
        actions.insert(player_id, Action::Right);
        world.step(actions, &mut rng);
    }

    println!("\n=== After moving right 4 times ===");
    println!("{}", world.render_entity_view(player_id));
}
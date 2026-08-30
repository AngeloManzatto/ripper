//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::fmt;
use rand::Rng;

//-----------------------------------------------------
// Types
//-----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cell {
    Empty
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Player,
    Enemy,
    Goal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    pub id: u32,
    pub kind: EntityKind,
    pub pos: (usize, usize),
}

pub struct GridConfig {
    pub agent_start: (usize, usize),
    pub goal_pos: (usize, usize),
    pub enemy_start: (usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
    Reset,
    Exit,
}


#[derive(Debug)]
pub struct StepResult {
    pub position: (
        usize, usize
    ),
    pub reward: f64,
    pub done: bool
}


//-----------------------------------------------------
// Grid
//-----------------------------------------------------
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub data: Vec<Vec<Cell>>,
    pub entities: Vec<Entity>,
    pub agent_start: (usize, usize),
    pub enemy_start:(usize, usize),
    pub goal_pos: (usize, usize),
    pub next_id: u32,
    pub player_id: Option<u32>,
    pub enemy_id: Option<u32>
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Cell::Empty => write!(f, "."),
        }
    }
}

//-----------------------------------------------------
// Spawn Entity 
//-----------------------------------------------------

pub fn spawn_entity(grid: &mut Grid, kind: EntityKind, pos:(usize, usize)) -> u32
{
    // Get next id for entity
    let id = grid.next_id;

    // Initialize Entity
    let entity: Entity = Entity { id: id, kind, pos };

    // Update next id
    grid.next_id += 1;

    // Add the new entity on the grid
    grid.entities.push(entity);

    id

}

//-----------------------------------------------------
// Create Grid
//-----------------------------------------------------
pub fn create_grid(width: usize, height: usize, config: GridConfig) -> Grid {

    // Initialize the grid with Empty cells
    let data = vec![vec![Cell::Empty; width]; height];
    
    // Initialize grid
    let mut grid: Grid = Grid {
        width,
        height,
        data,
        entities: Vec::new(),
        agent_start: config.agent_start,
        enemy_start: config.enemy_start,
        goal_pos: config.goal_pos,
        next_id: 0,
        player_id: None,
        enemy_id: None,
    };

    // Add entities to grid
    reset_grid(&mut grid);  // <- delegate all the spawning logic to reset_grid

    grid
}

//-----------------------------------------------------
// Reset Grid
//-----------------------------------------------------
pub fn reset_grid(grid: &mut Grid) {

    // Reset all cells to Empty
    for row in 0..grid.height {
        for col in 0..grid.width {
            grid.data[row][col] = Cell::Empty;
        }
    }

    // Reset id and entities
    grid.next_id = 0;
    grid.entities = Vec::new();

    // Add entities to grid
    let agent_start = grid.agent_start;
    let enemy_start = grid.enemy_start;
    let goal_pos = grid.goal_pos;
    let player_id = spawn_entity(grid, EntityKind::Player, agent_start);
    let enemy_id = spawn_entity(grid, EntityKind::Enemy, enemy_start);
    let goal_id = spawn_entity(grid, EntityKind::Goal, goal_pos);

    grid.player_id = Some(player_id);
    grid.enemy_id = Some(enemy_id);
}

//-----------------------------------------------------
// Print Grid
//-----------------------------------------------------
pub fn print_grid(grid: &Grid) {

    for row in 0..grid.height {
        for col in 0..grid.width {

            let found = grid.entities.iter().find(|e| e.pos == (row, col));

            match found {
                Some(entity) => {
                    match entity.kind {
                        EntityKind::Player => print!("P"),
                        EntityKind::Enemy  => print!("E"),
                        EntityKind::Goal   => print!("G"),
                    }
                }
                None => print!("."),
            }
        }
        println!();
    }
}

//-----------------------------------------------------
// Step Function
//-----------------------------------------------------

pub fn step(grid: &mut Grid, entity_id: u32, action: Action) -> StepResult {

    // Find the entity's CURRENT position by id.
    // Hint: grid.entities.iter().find(|e| e.id == entity_id) — same pattern as print_grid,
    // but this time you also need its .pos, and you'll need to handle the "not found" case
    // (for now, .expect("entity not found") is fine — a real recovery strategy can come later)
    let entity = grid.entities.iter().find(|e| e.id == entity_id).expect("entity not found");
    let current_pos: (usize, usize) = entity.pos;

    // Figure out the new row/col based on `action` — same logic as before,
    // but working with (usize, usize) tuples instead of Position now
    let new_pos: (usize, usize) = match action {
        Action::Up    => (if current_pos.0 > 0 {current_pos.0 - 1 } else {current_pos.0}, current_pos.1),
        Action::Down  => (if current_pos.0 < grid.height - 1 {current_pos.0 + 1 } else {current_pos.0}, current_pos.1),
        Action::Left  => (current_pos.0, if current_pos.1 > 0 {current_pos.1 - 1} else {current_pos.1}),
        Action::Right => (current_pos.0, if current_pos.1 < grid.width - 1 {current_pos.1 + 1} else {current_pos.1}),
        Action::Reset => current_pos,
        Action::Exit  => current_pos,
    };

    // Check if new_pos lands on any Goal-kind entity — search grid.entities again,
    // this time for kind == EntityKind::Goal && pos == new_pos
    let done: bool = grid.entities.iter().any(|e| e.kind == EntityKind::Goal && e.pos == new_pos);
    let reward: f64 = if done { 1.0 } else { 0.0 };

    // Move the entity
    if let Some(moving_entity) = grid.entities.iter_mut().find(|e| e.id == entity_id) {
        moving_entity.pos = new_pos;
    }

    StepResult {
        position: new_pos,
        reward,
        done,
    }
}


//-----------------------------------------------------
// Enemy Step
//-----------------------------------------------------
pub fn step_enemy(grid: &mut Grid)
{
    // Find enemy enetity
    let enemy_id = grid.enemy_id.expect("No enemy id set!");
    let enemy = grid.entities.iter().find(|e| e.id == enemy_id).expect("No enemy entity found!");
    let current_pos: (usize, usize) = enemy.pos;

    // Select random action (UP - 0, DOWN - 1, LEFT - 2, RIGHT - 3)
    let mut rng = rand::thread_rng();
    let direction = rng.gen_range(0..4); // 0, 1, 2, or 3

    // Computes new position
    let new_position = match direction {
        0 => (if current_pos.0 > 0 {current_pos.0 - 1 } else {current_pos.0}, current_pos.1),
        1 => (if current_pos.0 < grid.height - 1 {current_pos.0 + 1 } else {current_pos.0}, current_pos.1),
        2 => (current_pos.0, if current_pos.1 > 0 {current_pos.1 - 1} else {current_pos.1}),
        3 => (current_pos.0, if current_pos.1 < grid.width - 1 {current_pos.1 + 1} else {current_pos.1}),
        _ => current_pos
    };

    // Update enemy position
    if let Some(moving_entity) = grid.entities.iter_mut().find(|e| e.id == enemy_id) {
        moving_entity.pos = new_position
    };

}

//-----------------------------------------------------
// Check Collision 
//-----------------------------------------------------
pub fn check_collision(grid: &Grid) -> bool {
    let player_id = grid.player_id.expect("No player id set!");
    let enemy_id = grid.enemy_id.expect("No enemy id set!");

    let player_pos = grid.entities.iter().find(|e| e.id == player_id).expect("Player not found!").pos;
    let enemy_pos = grid.entities.iter().find(|e| e.id == enemy_id).expect("Enemy not found!").pos;

    player_pos == enemy_pos
}

//-----------------------------------------------------
// Get observation
//-----------------------------------------------------
pub fn get_observation(grid: &Grid) -> Vec<Vec<Vec<f64>>> {
    let mut observation = vec![
        vec![vec![0.0; grid.width]; grid.height]; // Player channel
        3 // 3 channels total
    ];

    for entity in &grid.entities {
        let channel = match entity.kind {
            EntityKind::Player => 0,
            EntityKind::Enemy  => 1,
            EntityKind::Goal   => 2,
        };
        observation[channel][entity.pos.0][entity.pos.1] = 1.0;
    }

    observation
}
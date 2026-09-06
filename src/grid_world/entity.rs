//-----------------------------------------------------
// Entities
//-----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Player,
    Enemy,
    Goal,
    Wall,
    Trap
}
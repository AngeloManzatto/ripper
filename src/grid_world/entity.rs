//-----------------------------------------------------
// Tile Kind
//-----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileKind {
    Free,
    Wall,
    Trap,
    Goal,
}

impl TileKind {
    pub fn to_char(&self) -> char {
        match self {
            TileKind::Free => '.',
            TileKind::Wall => '#',
            TileKind::Trap => 'T',
            TileKind::Goal => 'G',
        }
    }
    pub fn from_char(ch: char) -> Option<TileKind> {
        match ch {
            '.' => Some(TileKind::Free),
            '#' => Some(TileKind::Wall),
            'T' => Some(TileKind::Trap),
            'G' => Some(TileKind::Goal),
            _ => None,
        }
    }
}

//-----------------------------------------------------
// Entities kind
//-----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Player,
    Enemy,

}

impl EntityKind {
    pub fn to_char(&self) -> char {
        match self {
            EntityKind::Player => 'P',
            EntityKind::Enemy => 'E',
        }
    }
    pub fn from_char(ch: char) -> Option<EntityKind> {
        match ch {
            'P' => Some(EntityKind::Player),
            'E' => Some(EntityKind::Enemy),
            _ => None,
        }
    }
}

//-----------------------------------------------------
// Entity Status
//-----------------------------------------------------

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub enum EntityStatus {
    Alive,
    GoalReached,
    Trapped,
    Caught,
}

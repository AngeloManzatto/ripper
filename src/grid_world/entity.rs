//-----------------------------------------------------
// Entities
//-----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Player,
    Enemy,
    Goal,
    Wall,
    Trap,
}

//-----------------------------------------------------
// EntityKind <-> Char Mapping
//-----------------------------------------------------

impl EntityKind {
    pub fn to_char(&self) -> char {
        match self {
            EntityKind::Player => 'P',
            EntityKind::Enemy  => 'E',
            EntityKind::Goal   => 'G',
            EntityKind::Wall   => '#',
            EntityKind::Trap   => 'T',
        }
    }

    pub fn from_char(ch: char) -> Option<EntityKind> {
        match ch {
            'P' => Some(EntityKind::Player),
            'E' => Some(EntityKind::Enemy),
            'G' => Some(EntityKind::Goal),
            '#' => Some(EntityKind::Wall),
            'T' => Some(EntityKind::Trap),
            _ => None,
        }
    }
}

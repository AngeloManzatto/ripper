//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::collections::HashMap;

//-----------------------------------------------------
// Named trait
//-----------------------------------------------------

pub trait Named {
    fn name(&self) -> &str;
}

//-----------------------------------------------------
// Table
//-----------------------------------------------------

#[derive(Debug)]
pub struct Table<D> {
    defs: Vec<D>,
    by_name: HashMap<String, usize>,
}

impl<D: Named> Table<D> {
    pub fn new(defs: Vec<D>) -> Table<D> {
        let mut by_name = HashMap::new();
        for (position, def) in defs.iter().enumerate() {
            by_name.insert(def.name().to_string(), position);
        }
        Table { defs, by_name }
    }

    pub fn len(&self) -> usize {
        self.defs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    pub fn get(&self, position: usize) -> Option<&D> {
        self.defs.get(position)
    }

    pub fn position_of(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    pub fn get_by_name(&self, name: &str) -> Option<&D> {
        let position = self.position_of(name)?;
        self.get(position)
    }
}
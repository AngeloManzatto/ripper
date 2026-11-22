//-----------------------------------------------------
// libraries
//-----------------------------------------------------

use std::sync::Arc;

use crate::ripper::content::store::{Content, TerrainId};
use crate::ripper::core::grid::Grid;

use super::layout::{parse_layout, LayoutError};

//-----------------------------------------------------
// libraries
//-----------------------------------------------------

#[derive(Debug)]
pub struct Level {
    content: Arc<Content>,
    terrain: Grid<TerrainId>,
}

impl Level {
    
    pub fn from_layout(text: &str, content: Arc<Content>) -> Result<Level, Vec<LayoutError>> {
        let terrain = parse_layout(text, &content)?;
        Ok(Level { content, terrain })
    }

    pub fn content(&self) -> &Content {
        &self.content
    }

    pub fn terrain(&self) -> &Grid<TerrainId> {
        &self.terrain
    }
}
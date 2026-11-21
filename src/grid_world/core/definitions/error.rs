//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::fmt;
use std::path::PathBuf;

//-----------------------------------------------------
// Definition Error
//-----------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefinitionError {
    pub path: PathBuf,
    pub message: String,
}

//-----------------------
// New
//-----------------------

impl DefinitionError {
    pub fn new(path: PathBuf, message: String) -> DefinitionError {
        DefinitionError { path, message }
    }
}

//-----------------------
// Display
//-----------------------

impl fmt::Display for DefinitionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.message)
    }
}

//-----------------------------------------------------
// Format errors
//-----------------------------------------------------

pub fn format_errors(errors: &[DefinitionError]) -> String {
    let mut lines: Vec<String> = Vec::new();

    for error in errors {
        lines.push(error.to_string());
    }

    lines.join("\n")
}
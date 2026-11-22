//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::fmt;
use std::path::PathBuf;

//-----------------------------------------------------
// Definition Error
//-----------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackError {
    pub path: PathBuf,
    pub message: String,
}

//-----------------------
// New
//-----------------------

impl PackError {
    pub fn new(path: PathBuf, message: String) -> PackError {
        PackError { path, message }
    }
}

//-----------------------
// Display
//-----------------------

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.message)
    }
}

//-----------------------------------------------------
// Format errors
//-----------------------------------------------------

pub fn format_errors(errors: &[PackError]) -> String {
    let mut lines: Vec<String> = Vec::new();

    for error in errors {
        lines.push(error.to_string());
    }

    lines.join("\n")
}
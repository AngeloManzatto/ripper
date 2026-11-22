//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use std::path::PathBuf;
use super::super::error::{PackError, format_errors};

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn new_keeps_the_path_and_the_message() {

    let path   = PathBuf::from("base/wall.json");
    let message = String::from("bad glyph");
    let error   = PackError::new(path, message);

    assert_eq!(error.path, PathBuf::from("base/wall.json"));
    assert_eq!(error.message, "bad glyph");
}

#[test]
fn display_shows_the_path_then_the_message() {
    let error = PackError::new(PathBuf::from("base/wall.json"), String::from("bad glyph"));

    assert_eq!(error.to_string(), "base/wall.json: bad glyph");
}

#[test]
fn format_errors_puts_one_error_per_line() {
    let errors = vec![
        PackError::new(PathBuf::from("a.json"), String::from("first")),
        PackError::new(PathBuf::from("b.json"), String::from("second")),
    ];

    assert_eq!(format_errors(&errors), "a.json: first\nb.json: second");
}

#[test]
fn format_errors_of_nothing_is_empty() {
    assert_eq!(format_errors(&[]), "");
}
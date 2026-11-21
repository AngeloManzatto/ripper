//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[cfg(test)]
mod tests {

    use std::path::PathBuf;
    use crate::grid_world::core::definitions::error::DefinitionError;
    use crate::grid_world::core::definitions::error::format_errors;

    #[test]
    fn new_keeps_the_path_and_the_message() {

        let path   = PathBuf::from("base/wall.json");
        let message = String::from("bad glyph");
        let error   = DefinitionError::new(path, message);

        assert_eq!(error.path, PathBuf::from("base/wall.json"));
        assert_eq!(error.message, "bad glyph");
    }

    #[test]
    fn display_shows_the_path_then_the_message() {
        let error = DefinitionError::new(PathBuf::from("base/wall.json"), String::from("bad glyph"));

        assert_eq!(error.to_string(), "base/wall.json: bad glyph");
    }

    #[test]
    fn format_errors_puts_one_error_per_line() {
        let errors = vec![
            DefinitionError::new(PathBuf::from("a.json"), String::from("first")),
            DefinitionError::new(PathBuf::from("b.json"), String::from("second")),
        ];

        assert_eq!(format_errors(&errors), "a.json: first\nb.json: second");
    }

    #[test]
    fn format_errors_of_nothing_is_empty() {
        assert_eq!(format_errors(&[]), "");
    }

}
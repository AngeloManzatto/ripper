//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

use super::super::color::Color;

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn a_color_keeps_its_three_channels() {
    let color = Color::new(10, 20, 30);

    assert_eq!((color.r, color.g, color.b), (10, 20, 30));
}

#[test]
fn a_color_is_copied_not_moved() {
    let first = Color::new(1, 2, 3);
    let second = first;

    assert_eq!(first, second);
}

#[test]
fn a_hex_text_gives_its_channels() {
    assert_eq!(Color::from_hex("#ff0000"), Some(Color::new(255, 0, 0)));
    assert_eq!(Color::from_hex("#00ff00"), Some(Color::new(0, 255, 0)));
    assert_eq!(Color::from_hex("#0000ff"), Some(Color::new(0, 0, 255)));
}

#[test]
fn the_extremes_are_black_and_white() {
    assert_eq!(Color::from_hex("#000000"), Some(Color::new(0, 0, 0)));
    assert_eq!(Color::from_hex("#ffffff"), Some(Color::new(255, 255, 255)));
}

#[test]
fn upper_and_lower_case_digits_mean_the_same() {
    assert_eq!(Color::from_hex("#C8A050"), Color::from_hex("#c8a050"));
    assert_eq!(Color::from_hex("#c8a050"), Some(Color::new(200, 160, 80)));
}

#[test]
fn the_short_form_is_not_accepted() {
    assert_eq!(Color::from_hex("#fff"), None);
}

#[test]
fn a_missing_hash_is_not_accepted() {
    assert_eq!(Color::from_hex("ff0000"), None);
}

#[test]
fn text_of_the_wrong_length_is_not_accepted() {
    assert_eq!(Color::from_hex(""), None);
    assert_eq!(Color::from_hex("#"), None);
    assert_eq!(Color::from_hex("#ff00000"), None);
}

#[test]
fn a_character_that_is_not_a_hex_digit_is_not_accepted() {
    assert_eq!(Color::from_hex("#gg0000"), None);
    assert_eq!(Color::from_hex("#ff 000"), None);
}

#[test]
fn letters_wider_than_one_byte_are_refused_without_crashing() {
    assert_eq!("#aéaaa".len(), 7);
    assert_eq!(Color::from_hex("#aéaaa"), None);
}

#[test]
fn a_color_written_as_hex_is_padded_and_lowercase() {
    assert_eq!(Color::new(1, 2, 3).to_hex(), "#010203");
    assert_eq!(Color::new(255, 160, 0).to_hex(), "#ffa000");
}

#[test]
fn writing_and_reading_hex_leads_back_to_the_same_color() {
    for color in [Color::new(0, 0, 0), Color::new(200, 160, 80), Color::new(255, 255, 255)] {
        assert_eq!(Color::from_hex(&color.to_hex()), Some(color));
    }
}
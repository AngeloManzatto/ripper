//-----------------------------------------------------
// Tests
//-----------------------------------------------------

use std::path::PathBuf;

use super::super::read::{LoadedObject, LoadedPack};
use super::super::error::PackError;
use super::super::validate::{
    check_id,
    check_unique_ids,
    check_name_characters,
    check_name_matches_file,
    check_dependencies_exist,
    check_glyph,
    check_color,
    check_unique_glyphs,
    validate_object,
    validate_pack,
    load_order,
    validate
};

use super::super::validate::{check_terrain_count, MAX_TERRAINS};
use super::fixture::{object, pack};

//-----------------------------------------------------
// Checkers
//-----------------------------------------------------

fn check(glyph: Option<&str>) -> Vec<PackError> {
    let glyph = glyph.map(String::from);
    let mut errors = Vec::new();
    check_glyph(&glyph, &PathBuf::from("wall.json"), &mut errors);
    errors
}

fn check_id_in_base(id: &str) -> Vec<PackError> {
    let mut errors = Vec::new();
    check_id(id, "base", &PathBuf::from("wall.json"), &mut errors);
    errors
}

fn check_name(name: &str) -> Vec<PackError> {
    let mut errors = Vec::new();
    check_name_characters(name, &PathBuf::from("wall.json"), &mut errors);
    errors
}

fn check_file(name: &str, file: &str) -> Vec<PackError> {
    let mut errors = Vec::new();
    check_name_matches_file(name, &PathBuf::from(file), &mut errors);
    errors
}

//-----------------------------------------------------
// Tests
//-----------------------------------------------------

#[test]
fn one_visible_character_is_fine() {
    assert!(check(Some("#")).is_empty());
    assert!(check(Some(".")).is_empty());
}

#[test]
fn a_missing_glyph_is_fine() {
    assert!(check(None).is_empty());
}

#[test]
fn an_empty_glyph_is_an_error() {
    assert_eq!(check(Some("")).len(), 1);
}

#[test]
fn two_characters_are_an_error() {
    assert_eq!(check(Some("ab")).len(), 1);
}

#[test]
fn a_space_is_an_error() {
    assert_eq!(check(Some(" ")).len(), 1);
}

#[test]
fn a_control_character_is_an_error() {
    assert_eq!(check(Some("\t")).len(), 1);
}

#[test]
fn the_error_names_the_file_and_shows_the_glyph() {
    let errors = check(Some("ab"));

    assert_eq!(errors[0].path, PathBuf::from("wall.json"));
    assert!(errors[0].message.contains("\"ab\""));
}

#[test]
fn errors_are_added_to_the_ones_already_there() {
    let mut errors = vec![PackError::new(PathBuf::from("other.json"), String::from("earlier"))];

    check_glyph(&Some(String::from("ab")), &PathBuf::from("wall.json"), &mut errors);

    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0].message, "earlier");
}

#[test]
fn a_well_formed_id_of_the_same_pack_is_fine() {
    assert!(check_id_in_base("base:wall").is_empty());
}

#[test]
fn an_id_without_a_colon_is_an_error() {
    assert_eq!(check_id_in_base("wall").len(), 1);
}

#[test]
fn an_id_with_an_empty_part_is_an_error() {
    assert_eq!(check_id_in_base("base:").len(), 1);
    assert_eq!(check_id_in_base(":wall").len(), 1);
}

#[test]
fn an_id_with_two_colons_is_an_error() {
    assert_eq!(check_id_in_base("base:wall:x").len(), 1);
}

#[test]
fn an_id_of_another_pack_is_an_error_that_names_the_right_prefix() {
    let errors = check_id_in_base("ice:wall");

    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("`base:`"));
}

#[test]
fn the_error_names_the_file() {
    let errors = check_id_in_base("wall");

    assert_eq!(errors[0].path, PathBuf::from("wall.json"));
}

#[test]
fn lowercase_letters_digits_and_underscores_are_fine() {
    assert!(check_name("wall").is_empty());
    assert!(check_name("wall_2").is_empty());
    assert!(check_name("a1").is_empty());
}

#[test]
fn anything_else_is_an_error() {
    assert_eq!(check_name("Wall").len(), 1);
    assert_eq!(check_name("my wall").len(), 1);
    assert_eq!(check_name("wall-1").len(), 1);
    assert_eq!(check_name("w\u{e4}ll").len(), 1);
}

#[test]
fn an_empty_name_is_an_error() {
    assert_eq!(check_name("").len(), 1);
}

#[test]
fn the_error_names_the_file_and_shows_the_name() {
    let errors = check_name("Wall");

    assert_eq!(errors[0].path, PathBuf::from("wall.json"));
    assert!(errors[0].message.contains("`Wall`"));
}

#[test]
fn a_name_equal_to_the_file_name_is_fine() {
    assert!(check_file("wall", "wall.json").is_empty());
}

#[test]
fn the_folders_in_front_do_not_matter() {
    assert!(check_file("wall", "base/terrain/wall.json").is_empty());
}

#[test]
fn a_different_file_name_is_an_error_that_says_what_to_expect() {
    let errors = check_file("wall", "base/terrain/stone.json");

    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, PathBuf::from("base/terrain/stone.json"));
    assert!(errors[0].message.contains("`wall.json`"));
}

#[test]
fn capitals_count_as_a_difference() {
    assert_eq!(check_file("wall", "Wall.json").len(), 1);
}

#[test]
fn extra_dots_in_the_file_name_count_as_a_difference() {
    assert_eq!(check_file("wall", "wall.v2.json").len(), 1);
}

#[test]
fn a_good_object_has_no_errors() {
    let mut errors = Vec::new();

    validate_object(
        &pack("base"),
        &object("base:wall", Some("#"), "packs/base/terrain/wall.json"),
        &mut errors,
    );

    assert!(errors.is_empty());
}

#[test]
fn every_broken_rule_is_reported() {
    let mut errors = Vec::new();

    validate_object(
        &pack("base"),
        &object("ice:Wall", Some("ab"), "packs/base/terrain/stone.json"),
        &mut errors,
    );

    assert_eq!(errors.len(), 4);
}

#[test]
fn a_badly_shaped_id_is_one_error_not_many() {
    let mut errors = Vec::new();

    validate_object(
        &pack("base"),
        &object("wall", Some("#"), "packs/base/terrain/wall.json"),
        &mut errors,
    );

    assert_eq!(errors.len(), 1);
}

// 

fn pack_in(folder: &str, name: &str) -> LoadedPack {
    let mut p = pack(name);
    p.dir = PathBuf::from("packs").join(folder);
    p
}

#[test]
fn a_pack_in_a_folder_of_the_same_name_is_fine() {
    let mut errors = Vec::new();

    validate_pack(&pack_in("base", "base"), &mut errors);

    assert!(errors.is_empty());
}

#[test]
fn a_pack_in_a_folder_of_another_name_is_an_error() {
    let mut errors = Vec::new();

    validate_pack(&pack_in("ice", "base"), &mut errors);

    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, PathBuf::from("packs").join("ice").join("pack.json"));
    assert!(errors[0].message.contains("folder called `base`"));
}

#[test]
fn the_supported_format_version_and_none_are_fine() {
    let mut errors = Vec::new();
    let mut versioned = pack("base");
    versioned.raw.format_version = Some(1);

    validate_pack(&versioned, &mut errors);
    validate_pack(&pack("base"), &mut errors);

    assert!(errors.is_empty());
}

#[test]
fn another_format_version_is_an_error_that_shows_it() {
    let mut errors = Vec::new();
    let mut versioned = pack("base");
    versioned.raw.format_version = Some(2);

    validate_pack(&versioned, &mut errors);

    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("format_version 2"));
}

#[test]
fn an_object_with_another_format_version_is_an_error() {
    let mut errors = Vec::new();
    let mut o = object("base:wall", Some("#"), "packs/base/terrain/wall.json");
    o.terrain.format_version = Some(2);

    validate_object(&pack("base"), &o, &mut errors);

    assert_eq!(errors.len(), 1);
}

fn with_objects(mut p: LoadedPack, objects: Vec<LoadedObject>) -> LoadedPack {
p.objects = objects;
p
}

#[test]
fn different_ids_are_fine() {
let mut errors = Vec::new();
let packs = vec![with_objects(
    pack("base"),
    vec![
        object("base:wall", Some("#"), "base/wall.json"),
        object("base:empty", Some("."), "base/empty.json"),
    ],
)];

check_unique_ids(&packs, &mut errors);

assert!(errors.is_empty());
}

#[test]
fn no_packs_is_fine() {
let mut errors = Vec::new();

check_unique_ids(&[], &mut errors);

assert!(errors.is_empty());
}

#[test]
fn the_same_id_twice_is_an_error_on_the_second_file() {
let mut errors = Vec::new();
let packs = vec![with_objects(
    pack("base"),
    vec![
        object("base:wall", Some("#"), "base/wall.json"),
        object("base:wall", Some("#"), "base/wall_copy.json"),
    ],
)];

check_unique_ids(&packs, &mut errors);

    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, PathBuf::from("base/wall_copy.json"));
    assert!(errors[0].message.contains("base/wall.json"));
}

#[test]
fn the_same_id_in_two_packs_is_an_error() {
    let mut errors = Vec::new();
    let packs = vec![
        with_objects(pack("base"), vec![object("base:wall", Some("#"), "base/wall.json")]),
        with_objects(pack("ice"), vec![object("base:wall", Some("#"), "ice/wall.json")]),
    ];

    check_unique_ids(&packs, &mut errors);

    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, PathBuf::from("ice/wall.json"));
}

#[test]
fn three_copies_give_two_errors_both_pointing_at_the_first() {
    let mut errors = Vec::new();
    let packs = vec![with_objects(
        pack("base"),
        vec![
            object("base:wall", Some("#"), "a.json"),
            object("base:wall", Some("#"), "b.json"),
            object("base:wall", Some("#"), "c.json"),
        ],
    )];

    check_unique_ids(&packs, &mut errors);

    assert_eq!(errors.len(), 2);
    assert!(errors[0].message.contains("a.json"));
    assert!(errors[1].message.contains("a.json"));
}

fn depending_on(mut p: LoadedPack, names: &[&str]) -> LoadedPack {
    let mut list = Vec::new();
    for name in names {
        list.push(String::from(*name));
    }
    p.raw.depends_on = list;
    p
}

#[test]
fn dependencies_that_exist_are_fine() {
    let mut errors = Vec::new();
    let packs = vec![pack("base"), depending_on(pack("ice"), &["base"])];

    check_dependencies_exist(&packs, &mut errors);

    assert!(errors.is_empty());
}

#[test]
fn packs_without_dependencies_are_fine() {
    let mut errors = Vec::new();

    check_dependencies_exist(&[pack("base")], &mut errors);

    assert!(errors.is_empty());
}

#[test]
fn a_missing_dependency_is_an_error_that_names_both_packs() {
    let mut errors = Vec::new();
    let packs = vec![pack("base"), depending_on(pack("ice"), &["ghost"])];

    check_dependencies_exist(&packs, &mut errors);

    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, PathBuf::from("packs").join("ice").join("pack.json"));
    assert!(errors[0].message.contains("`ice`"));
    assert!(errors[0].message.contains("`ghost`"));
}

#[test]
fn every_missing_dependency_is_reported() {
    let mut errors = Vec::new();
    let packs = vec![
        depending_on(pack("ice"), &["ghost", "base", "phantom"]),
        depending_on(pack("fire"), &["ghost"]),
    ];

    check_dependencies_exist(&packs, &mut errors);

    assert_eq!(errors.len(), 4);
}

fn names_in_order(packs: &[LoadedPack], order: &[usize]) -> Vec<String> {
    let mut names = Vec::new();
    for index in order {
        names.push(packs[*index].raw.name.clone());
    }
    names
}

#[test]
fn packs_without_dependencies_load_in_name_order() {
    let mut errors = Vec::new();
    let packs = vec![pack("b"), pack("a"), pack("c")];

    let order = load_order(&packs, &mut errors);

    assert!(errors.is_empty());
    assert_eq!(names_in_order(&packs, &order), vec!["a", "b", "c"]);
}

#[test]
fn a_pack_loads_after_what_it_depends_on() {
    let mut errors = Vec::new();
    let packs = vec![depending_on(pack("a"), &["z"]), pack("z")];

    let order = load_order(&packs, &mut errors);

    assert_eq!(names_in_order(&packs, &order), vec!["z", "a"]);
}

#[test]
fn a_chain_and_a_tie_are_ordered_by_dependency_then_name() {
    let mut errors = Vec::new();
    let packs = vec![
        depending_on(pack("ice"), &["base"]),
        depending_on(pack("fire"), &["base"]),
        pack("base"),
        depending_on(pack("steam"), &["fire", "ice"]),
    ];

    let order = load_order(&packs, &mut errors);

    assert!(errors.is_empty());
    assert_eq!(names_in_order(&packs, &order), vec!["base", "fire", "ice", "steam"]);
}

#[test]
fn two_packs_depending_on_each_other_are_both_reported() {
    let mut errors = Vec::new();
    let packs = vec![depending_on(pack("a"), &["b"]), depending_on(pack("b"), &["a"])];

    let order = load_order(&packs, &mut errors);

    assert!(order.is_empty());
    assert_eq!(errors.len(), 2);
}

#[test]
fn a_pack_depending_on_itself_is_reported() {
    let mut errors = Vec::new();
    let packs = vec![depending_on(pack("a"), &["a"])];

    load_order(&packs, &mut errors);

    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("`a`"));
}

#[test]
fn a_cycle_does_not_stop_the_other_packs_from_being_ordered() {
    let mut errors = Vec::new();
    let packs = vec![
        depending_on(pack("a"), &["b"]),
        depending_on(pack("b"), &["a"]),
        pack("x"),
    ];

    let order = load_order(&packs, &mut errors);

    assert_eq!(names_in_order(&packs, &order), vec!["x"]);
    assert_eq!(errors.len(), 2);
}

#[test]
fn a_pack_depending_on_a_cycle_is_reported_too() {
    let mut errors = Vec::new();
    let packs = vec![
        depending_on(pack("a"), &["b"]),
        depending_on(pack("b"), &["a"]),
        depending_on(pack("c"), &["a"]),
    ];

    load_order(&packs, &mut errors);

    assert_eq!(errors.len(), 3);
}

#[test]
fn an_unknown_dependency_is_left_to_the_other_rule() {
    let mut errors = Vec::new();
    let packs = vec![depending_on(pack("a"), &["ghost"])];

    let order = load_order(&packs, &mut errors);

    assert!(errors.is_empty());
    assert_eq!(order, vec![0]);
}

//

#[test]
fn a_good_set_of_packs_passes_and_gets_an_order() {
    let mut errors = Vec::new();
    let packs = vec![
        with_objects(
            depending_on(pack("ice"), &["base"]),
            vec![object("ice:snow", Some("*"), "packs/ice/terrain/snow.json")],
        ),
        with_objects(
            pack("base"),
            vec![
                object("base:wall", Some("#"), "packs/base/terrain/wall.json"),
                object("base:empty", Some("."), "packs/base/terrain/empty.json"),
            ],
        ),
    ];

    let order = validate(&packs, &mut errors);

    assert!(errors.is_empty());
    assert_eq!(names_in_order(&packs, &order), vec!["base", "ice"]);
}

#[test]
fn problems_of_every_kind_are_reported_together() {
    let mut errors = Vec::new();
    let packs = vec![
        with_objects(
            pack("base"),
            vec![object("base:wall", Some("#"), "packs/base/terrain/wall.json")],
        ),
        with_objects(
            depending_on(pack_in("icy", "ice"), &["ghost"]),
            vec![object("ice:Snow", Some("ab"), "packs/icy/terrain/snow.json")],
        ),
    ];

    let order = validate(&packs, &mut errors);

    assert_eq!(errors.len(), 5);
    assert_eq!(order, vec![0, 1]);
}

#[test]
fn different_glyphs_are_fine_and_missing_glyphs_never_clash() {
    let mut errors = Vec::new();
    let packs = vec![with_objects(
        pack("base"),
        vec![
            object("base:wall", Some("#"), "base/wall.json"),
            object("base:empty", Some("."), "base/empty.json"),
            object("base:a", None, "base/a.json"),
            object("base:b", None, "base/b.json"),
        ],
    )];

    check_unique_glyphs(&packs, &mut errors);

    assert!(errors.is_empty());
}

#[test]
fn the_same_glyph_twice_is_an_error_on_the_second_file() {
    let mut errors = Vec::new();
    let packs = vec![with_objects(
        pack("base"),
        vec![
            object("base:wall", Some("#"), "base/wall.json"),
            object("base:stone", Some("#"), "base/stone.json"),
        ],
    )];

    check_unique_glyphs(&packs, &mut errors);

    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, PathBuf::from("base/stone.json"));
    assert!(errors[0].message.contains("`#`"));
    assert!(errors[0].message.contains("base/wall.json"));
}

#[test]
fn the_same_glyph_in_two_packs_is_an_error() {
    let mut errors = Vec::new();
    let packs = vec![
        with_objects(pack("base"), vec![object("base:wall", Some("#"), "base/wall.json")]),
        with_objects(pack("ice"), vec![object("ice:wall", Some("#"), "ice/wall.json")]),
    ];

    check_unique_glyphs(&packs, &mut errors);

    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, PathBuf::from("ice/wall.json"));
}

#[test]
fn validate_runs_the_glyph_rule() {
    let mut errors = Vec::new();
    let packs = vec![with_objects(
        pack("base"),
        vec![
            object("base:wall", Some("#"), "packs/base/terrain/wall.json"),
            object("base:stone", Some("#"), "packs/base/terrain/stone.json"),
        ],
    )];

    validate(&packs, &mut errors);

    assert_eq!(errors.len(), 1);
}

#[test]
fn the_limit_of_terrains_is_enforced() {
    let root = PathBuf::from("packs");
    let mut errors: Vec<PackError> = Vec::new();

    check_terrain_count(MAX_TERRAINS, &root, &mut errors);
    assert!(errors.is_empty());

    check_terrain_count(MAX_TERRAINS + 1, &root, &mut errors);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, root);
    assert!(errors[0].message.contains("too many terrains"));
}

fn check_color_text(field: &str, text: &str) -> Vec<PackError> {
    let mut errors = Vec::new();
    check_color(field, text, &PathBuf::from("wall.json"), &mut errors);
    errors
}

#[test]
fn a_good_color_is_fine() {
    assert!(check_color_text("fg_color", "#c8a050").is_empty());
}

#[test]
fn upper_case_digits_are_fine() {
    assert!(check_color_text("bg_color", "#C8A050").is_empty());
}

#[test]
fn a_color_name_is_not_a_color() {
    assert_eq!(check_color_text("fg_color", "red").len(), 1);
}

#[test]
fn the_short_form_is_refused() {
    assert_eq!(check_color_text("fg_color", "#fff").len(), 1);
}

#[test]
fn a_missing_hash_is_refused() {
    assert_eq!(check_color_text("fg_color", "c8a050").len(), 1);
}

#[test]
fn a_digit_that_is_not_hex_is_refused() {
    assert_eq!(check_color_text("fg_color", "#gg0000").len(), 1);
}

#[test]
fn an_empty_text_is_refused() {
    assert_eq!(check_color_text("bg_color", "").len(), 1);
}

#[test]
fn wide_letters_are_refused_without_crashing() {
    assert_eq!(check_color_text("fg_color", "#aéaaa").len(), 1);
}

#[test]
fn the_message_names_the_field_and_the_text() {
    let errors = check_color_text("bg_color", "red");

    assert!(errors[0].message.contains("bg_color"));
    assert!(errors[0].message.contains("\"red\""));
}

#[test]
fn the_color_error_points_at_the_file() {
    assert_eq!(check_color_text("fg_color", "red")[0].path, PathBuf::from("wall.json"));
}

#[test]
fn bad_colors_are_reported_through_validate_object() {
    let mut errors = Vec::new();
    let mut broken = object("base:wall", Some("#"), "packs/base/terrain/wall.json");
    broken.terrain.fg_color = String::from("red");
    broken.terrain.bg_color = String::from("#12");

    validate_object(&pack("base"), &broken, &mut errors);

    assert_eq!(errors.len(), 2);
}
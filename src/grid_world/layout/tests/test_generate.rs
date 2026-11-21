//-----------------------------------------------------
// Imports
//-----------------------------------------------------

use rand::{rngs::StdRng, SeedableRng};

//-----------------------------------------------------
// Test
//-----------------------------------------------------

#[cfg(test)]
mod tests {

    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn print_grid(grid: &Vec<Vec<char>>) {
        for row in grid {
            let line: String = row.iter().collect();
            println!("{}", line);
        }
    }

    #[test]
    fn test_add_random_walls_to_grid_deterministic() {
        let mut rng = StdRng::seed_from_u64(42);

        let mut grid = generate_default_grid(6, 6);
        grid = pad_grid_with_walls(grid, 1);
        grid = add_random_walls_to_grid(grid, 0.3, &mut rng);

        // Print grid
        print_grid(&grid);

        // same seed, same density, same grid size → always the exact same output
        let expected = vec![
            vec!['#', '#', '#', '#', '#', '#'],
            vec!['#', '.', '.', '#', '#', '#'],
            vec!['#', '.', '.', '#', '.', '#'],
            vec!['#', '#', '.', '#', '.', '#'],
            vec!['#', '.', '.', '#', '.', '#'],
            vec!['#', '#', '#', '#', '#', '#'],
        ];

        assert_eq!(grid, expected);
    }

    #[test]
    fn test_generate_default_layout() {
        // Default random generator
        let mut rng = StdRng::seed_from_u64(42);

        // Default config
        let config = GenerationConfig::default();

        // Generate an empty grid
        let layout = generate_layout(&config, &mut rng).expect("generation should succeed");

        println!("{:?}", layout);

        assert_eq!(
            layout,
            "################\n##.#....#......#\n#.......#....###\n#..............#\n##..#.E..#.....#\n#.#........#...#\n#.....#...#....#\n#.#....#.....#.#\n#..#....#P.#...#\n###.#...##....G#\n#..#.#.#.#.....#\n#..#..##.......#\n#.#.....###....#\n#......##......#\n#..#........##.#\n################"
        );
    }

    #[test]
    fn test_generate_layout_with_traps() {
        // Default random generator
        let mut rng = StdRng::seed_from_u64(42);

        // Default config
        let mut config = GenerationConfig::default();
        config.num_traps = 2;

        // Generate an empty grid
        let layout = generate_layout(&config, &mut rng).expect("generation should succeed");

        println!("{:?}", layout);

        assert_eq!(
            layout,
            "################\n##.#...E#......#\n#....T..#....###\n#..............#\n##..#....#.....#\n#.#........#...#\n#.....#...#....#\n#.#....#.....#.#\n#..#....#P.#...#\n###.#...##....G#\n#..#.#.#.#..T..#\n#..#..##.......#\n#.#.....###....#\n#......##......#\n#..#........##.#\n################"
        );
    }
}

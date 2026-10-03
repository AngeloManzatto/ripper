//-----------------------------------------------------
// Layout to Chars
//-----------------------------------------------------

pub fn layout_to_grid(layout: &str) -> Vec<Vec<char>> {
    layout.lines().map(|line| line.chars().collect()).collect()
}

//-----------------------------------------------------
// Chars to Layout
//-----------------------------------------------------

pub fn grid_to_layout(grid: &Vec<Vec<char>>) -> String {
    grid.iter()
        .map(|row| row.iter().collect::<String>())
        .collect::<Vec<String>>()
        .join("\n")
}

//-----------------------------------------------------
// Find Char
//-----------------------------------------------------

pub fn find_char(grid: &[Vec<char>], target: char) -> Option<(usize, usize)> {
    for (row, line) in grid.iter().enumerate() {
        for (col, &ch) in line.iter().enumerate() {
            if ch == target {
                return Some((row, col));
            }
        }
    }
    None
}

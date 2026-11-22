//-----------------------------------------------------
// Libraries
//-----------------------------------------------------

//-----------------------------------------------------
// Color
//-----------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

//-----------------------------------------------------
// Color Imp
//-----------------------------------------------------

impl Color {
    pub fn new(r: u8, g: u8, b: u8) -> Color {
        Color { r, g, b }
    }

    pub fn from_hex(text: &str) -> Option<Color> {
        if !text.starts_with('#') || text.len() != 7 {
            return None;
        }
        if !text[1..].chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }

        let r = u8::from_str_radix(&text[1..3], 16).ok()?;
        let g = u8::from_str_radix(&text[3..5], 16).ok()?;
        let b = u8::from_str_radix(&text[5..7], 16).ok()?;

        Some(Color { r, g, b })
    }

    pub fn to_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}
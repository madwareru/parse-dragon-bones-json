use serde::Deserialize;

#[derive(Copy, Clone, Deserialize, Default, Debug)]
pub struct Rect {
    #[serde(default)]
    pub x: f32,
    #[serde(default)]
    pub y: f32,
    #[serde(default)]
    pub width: f32,
    #[serde(default)]
    pub height: f32,
}

#[derive(Copy, Clone, Deserialize, Default, Debug)]
pub struct Point {
    #[serde(default)]
    pub x: f32,
    #[serde(default)]
    pub y: f32,
}

#[derive(Copy, Clone, Deserialize, Debug)]
pub enum BlendMode {
    #[serde(rename = "normal")]
    Normal,
    #[serde(rename = "add")]
    Add,
    #[serde(rename = "alpha")]
    Alpha,
    #[serde(rename = "darken")]
    Darken,
    #[serde(rename = "difference")]
    Difference,
    #[serde(rename = "erase")]
    Erase,
    #[serde(rename = "hardlight")]
    HardLight,
    #[serde(rename = "invert")]
    Invert,
    #[serde(rename = "layer")]
    Layer,
    #[serde(rename = "lighten")]
    Lighten,
    #[serde(rename = "multiply")]
    Multiply,
    #[serde(rename = "overlay")]
    Overlay,
    #[serde(rename = "screen")]
    Screen,
    #[serde(rename = "subtract")]
    Subtract
}
impl Default for BlendMode {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Copy, Clone, Deserialize, Debug)]
pub struct ColorTransform {
    #[serde(rename = "aM")]
    #[serde(default = "default_100")]
    pub alpha_multiplier: i32,
    #[serde(rename = "rM")]
    #[serde(default = "default_100")]
    pub red_multiplier: i32,
    #[serde(rename = "gM")]
    #[serde(default = "default_100")]
    pub green_multiplier: i32,
    #[serde(rename = "bM")]
    #[serde(default = "default_100")]
    pub blue_multiplier: i32,
    #[serde(rename = "aO")]
    #[serde(default)]
    pub alpha_offset: i32,
    #[serde(rename = "rO")]
    #[serde(default)]
    pub red_offset: i32,
    #[serde(rename = "gO")]
    #[serde(default)]
    pub green_offset: i32,
    #[serde(rename = "bO")]
    #[serde(default)]
    pub blue_offset: i32,
}
impl Default for ColorTransform {
    fn default() -> Self {
        Self {
            alpha_multiplier: 100,
            red_multiplier: 100,
            green_multiplier: 100,
            blue_multiplier: 100,
            alpha_offset: 0,
            red_offset: 0,
            green_offset: 0,
            blue_offset: 0
        }
    }
}

// Do not do IK constraints for now
// #[derive(Clone, Deserialize)]
// pub struct IkConstraint {
//     #[serde(default)]
//     bone: String,
//     #[serde(default)]
//     target: String,
// }

pub(crate) fn default_true() -> bool { true }
pub(crate) fn default_one() -> f32 { 1.0 }
pub(crate) fn default_no_easing() -> f32 { -2.0 }
pub(crate) fn default_100() -> i32 { 100 }
pub(crate) fn default_play() -> String { "play".into() }
pub(crate) fn default_name() -> String { "default".into() }
pub(crate) fn default_none_parent() -> String { "__none__".into() }
pub(crate) fn default_pivot() -> Point { Point { x: 0.5, y: 0.5 } }
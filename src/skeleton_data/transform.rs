use serde::Deserialize;
use crate::shared_types::default_one;

#[derive(Copy, Clone, Deserialize, Debug)]
pub struct RawTransform {
    #[serde(default)]
    pub x: f32,

    #[serde(default)]
    pub y: f32,

    #[serde(default)]
    pub skew: f32,

    #[serde(default)]
    #[serde(rename = "rotate")]
    pub rotation: f32,

    #[serde(default)]
    #[serde(rename = "skX")]
    pub skew_x: f32,

    #[serde(default)]
    #[serde(rename = "skY")]
    pub skew_y: f32,

    #[serde(default = "default_one")]
    #[serde(rename = "scX")]
    pub scale_x: f32,

    #[serde(default = "default_one")]
    #[serde(rename = "scY")]
    pub scale_y: f32,
}
impl Default for RawTransform {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            skew: 0.0,
            rotation: 0.0,
            skew_x: 0.0,
            skew_y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PurifiedTransform {
    pub x: f32,
    pub y: f32,
    pub rotation: f32,
    pub skew: f32,
    pub scale_x: f32,
    pub scale_y: f32,
}
impl From<RawTransform> for PurifiedTransform {
    fn from(initial: RawTransform) -> Self {
        let (rotation, skew) = if initial.rotation != 0.0 || initial.skew != 0.0 {
            (
                normalize_radian(initial.rotation.to_radians()),
                normalize_radian(initial.skew.to_radians())
            )
        } else if initial.skew_x != 0.0 || initial.skew_y != 0.0 {
            let rot = normalize_radian(initial.skew_y.to_radians());
            (
                rot,
                normalize_radian(initial.skew_x.to_radians()) - rot
            )
        } else {
            (0.0, 0.0)
        };
        Self {
            x: initial.x,
            y: initial.y,
            scale_x: initial.scale_x,
            scale_y: initial.scale_y,
            rotation,
            skew
        }
    }
}

fn normalize_radian(value: f32) -> f32 {
    let value = (value + std::f32::consts::PI) % (std::f32::consts::TAU);
    if value > 0.0 {
        value - std::f32::consts::PI
    } else {
        value + std::f32::consts::PI
    }
}
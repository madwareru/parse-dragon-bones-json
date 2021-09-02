use serde::Deserialize;
use crate::shared_types::{default_true, default_none_parent};

#[derive(Clone, Deserialize, Debug)]
pub struct RawBone {
    pub name: String,

    #[serde(default = "default_none_parent")]
    pub parent: String,

    #[serde(rename = "inheritTranslation")]
    #[serde(default = "default_true")]
    pub inherit_translation: bool,

    #[serde(rename = "inheritRotation")]
    #[serde(default = "default_true")]
    pub inherit_rotation: bool,

    #[serde(rename = "inheritScale")]
    #[serde(default = "default_true")]
    pub inherit_scale: bool,

    #[serde(rename = "inheritReflection")]
    #[serde(default = "default_true")]
    pub inherit_reflection: bool,

    #[serde(default)]
    pub length: f32,

    #[serde(default)]
    pub transform: super::transform::RawTransform
}
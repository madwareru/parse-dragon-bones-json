use serde::Deserialize;

#[derive(Clone, Deserialize, Debug)]
pub struct RawSlot {
    #[serde(rename = "displayIndex")]
    #[serde(default)]
    pub display_id: i32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub parent: String,
    #[serde(default)]
    #[serde(rename = "blendMode")]
    pub blend_mode: crate::shared_types::BlendMode,
    #[serde(default)]
    #[serde(rename = "color")]
    pub color_transform: crate::shared_types::ColorTransform,
    #[serde(default)]
    pub actions: Vec<super::actions::RawActionData>
}
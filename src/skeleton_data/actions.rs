use serde::Deserialize;
use crate::shared_types::default_play;

#[derive(Clone, Deserialize, Debug)]
pub struct RawActionData {
    #[serde(rename = "gotoAndPlay")]
    #[serde(default)]
    pub goto_and_play: String,
    #[serde(rename = "type")]
    #[serde(default = "default_play")]
    pub action_type: String,
    #[serde(default)]
    pub name: String,
    #[serde(rename = "bone")]
    #[serde(default)]
    pub bone_name: String,
    #[serde(rename = "slot")]
    #[serde(default)]
    pub slot_name: String,
    #[serde(rename = "ints")]
    #[serde(default)]
    pub user_ints: Vec<i32>,
    #[serde(rename = "floats")]
    #[serde(default)]
    pub user_floats: Vec<f32>,
    #[serde(rename = "strings")]
    #[serde(default)]
    pub user_strings: Vec<String>
}
use serde::Deserialize;
use crate::shared_types::{default_one, default_true};

#[derive(Clone, Deserialize, Debug)]
pub struct IkInfo {
    #[serde(rename = "bendPositive")]
    #[serde(default = "default_true")]
    pub bend_positive: bool,
    #[serde(rename = "chain")]
    #[serde(default)]
    pub chain_length: usize,
    #[serde(default = "default_one")]
    pub weight: f32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub bone: String,
    #[serde(default)]
    pub target: String
}
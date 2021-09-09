pub mod armature;
pub mod bone;
pub mod transform;
pub mod slot;
pub mod actions;
pub mod animation;
pub mod skin;
pub mod ik;

use serde::Deserialize;

#[derive(Clone, Deserialize, Debug)]
pub struct RawSkeletonData {
    #[serde(rename = "frameRate")]
    pub frame_rate: u32,

    pub name: String,

    pub version: String,

    #[serde(rename = "compatibleVersion")]
    pub compatible_version: String,

    #[serde(rename = "armature")]
    #[serde(default)]
    pub armatures: Vec<crate::skeleton_data::armature::RawArmatureData>,
}
use serde::Deserialize;

#[derive(Clone, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum RawArmatureData {
    // None, MovieClip and Stage are unsupported, if there would be any of these we expect to fail on parsing
     Armature {
        #[serde(rename = "frameRate")]
        frame_rate: u32,
        name: String,

        #[serde(rename = "aabb")]
        #[serde(default)]
        aa_bb: crate::shared_types::Rect,

        #[serde(rename = "bone")]
        #[serde(default)]
        bones: Vec<super::bone::RawBone>,

        #[serde(default)]
        ik: Vec<super::ik::IkInfo>,

        #[serde(rename = "slot")]
        #[serde(default)]
        slots: Vec<super::slot::RawSlot>,

        #[serde(rename = "skin")]
        #[serde(default)]
        skins: Vec<super::skin::RawSkinData>,

        #[serde(rename = "animation")]
        #[serde(default)]
        animations: Vec<super::animation::RawAnimationData>,

        #[serde(rename = "defaultActions")]
        #[serde(default)]
        default_actions: Vec<crate::skeleton_data::actions::RawActionData>,

        #[serde(default)]
        actions: Vec<crate::skeleton_data::actions::RawActionData>,
    }
}
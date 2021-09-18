use serde::Deserialize;
use crate::shared_types::{default_one, default_one_frame, default_name, default_no_easing, ColorTransform};

#[derive(Clone, Deserialize, Debug)]
pub struct RawAnimationData {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default)]
    pub duration: u32,

    #[serde(rename = "playTimes")]
    #[serde(default)]
    pub play_times: u32,
    #[serde(rename = "fadeInTime")]
    #[serde(default)]
    pub fade_in_time: f32,
    #[serde(default = "default_one")]
    pub scale: f32,
    #[serde(rename = "frame")]
    #[serde(default)]
    pub general_timeline: Vec<RawGeneralFrame>,
    #[serde(rename = "zOrder")]
    #[serde(default)]
    pub z_order_timeline: Vec<RawZOrderFrame>,
    #[serde(rename = "slot")]
    #[serde(default)]
    pub slot_timelines: Vec<RawSlotTimeline>,
    //we are not support animation of FFD and IK parameters yet
    #[serde(rename = "bone")]
    #[serde(default)]
    pub bone_timelines: Vec<RawBoneTimeline>
}

#[derive(Clone, Deserialize, Debug)]
pub struct RawTranslationFrame {
    #[serde(default = "default_one_frame")]
    pub duration: u32,
    #[serde(default)]
    pub curve: Vec<f32>,
    #[serde(default = "default_no_easing")]
    pub tween_easing: f32,
    #[serde(default)]
    pub x: f32,
    #[serde(default)]
    pub y: f32,
}

#[derive(Clone, Deserialize, Debug)]
pub struct RawRotationFrame {
    #[serde(default = "default_one_frame")]
    pub duration: u32,
    #[serde(default)]
    pub curve: Vec<f32>,
    #[serde(default = "default_no_easing")]
    pub tween_easing: f32,
    #[serde(default)]
    pub rotate: f32,
    #[serde(default)]
    pub clockwise: u32,
}

#[derive(Clone, Deserialize, Debug)]
pub struct RawScaleFrame {
    #[serde(default = "default_one_frame")]
    pub duration: u32,
    #[serde(default)]
    pub curve: Vec<f32>,
    #[serde(default = "default_no_easing")]
    pub tween_easing: f32,
    #[serde(default = "default_one")]
    pub x: f32,
    #[serde(default = "default_one")]
    pub y: f32,
}

#[derive(Clone, Deserialize, Debug)]
pub struct RawGeneralFrame {
    #[serde(default = "default_one_frame")]
    pub duration: u32,
    #[serde(default)]
    pub curve: Vec<f32>,
    #[serde(default = "default_no_easing")]
    pub tween_easing: f32,
    #[serde(default)]
    pub transform: super::transform::RawTransform,
    #[serde(rename = "tweenRotate")]
    #[serde(default)]
    pub clockwise: u32,
    #[serde(default)]
    pub event: Vec<super::actions::RawActionData>,
    #[serde(default)]
    pub sound: Vec<super::actions::RawActionData>,
    #[serde(default)]
    pub action: Vec<super::actions::RawActionData>,
    #[serde(default)]
    pub events: Vec<super::actions::RawActionData>,
    #[serde(default)]
    pub actions: Vec<super::actions::RawActionData>,
}

#[derive(Clone, Deserialize, Debug)]
pub struct RawZOrderFrame {
    #[serde(default = "default_one_frame")]
    pub duration: u32,
    #[serde(default)]
    pub curve: Vec<f32>,
    #[serde(default = "default_no_easing")]
    pub tween_easing: f32,
    #[serde(rename = "zOrder")]
    #[serde(default)]
    pub z_order: Vec<i32>,
}

#[derive(Clone, Deserialize, Debug)]
pub struct RawSlotTimeline {
    pub name: String,
    #[serde(rename = "displayFrame")]
    #[serde(default)]
    pub display_frames: Vec<RawDisplayFrame>,
    #[serde(rename = "colorFrame")]
    #[serde(default)]
    pub colors: Vec<RawColorFrame>
}

#[derive(Clone, Deserialize, Debug)]
pub struct RawDisplayFrame {
    #[serde(default = "default_one_frame")]
    pub duration: u32,
    #[serde(default)]
    pub value: i32
}

#[derive(Clone, Deserialize, Debug)]
pub struct RawColorFrame {
    #[serde(default = "default_one_frame")]
    pub duration: u32,
    #[serde(default)]
    pub curve: Vec<f32>,
    #[serde(default = "default_no_easing")]
    pub tween_easing: f32,
    #[serde(rename = "value")]
    #[serde(default)]
    pub color: ColorTransform
}

#[derive(Clone, Deserialize, Debug)]
pub struct RawBoneTimeline {
    #[serde(rename = "name")]
    #[serde(default)]
    pub bone_name: String,
    #[serde(rename = "translateFrame")]
    #[serde(default)]
    pub translate_frames: Vec<RawTranslationFrame>,
    #[serde(rename = "scaleFrame")]
    #[serde(default)]
    pub scale_frames: Vec<RawScaleFrame>,
    #[serde(rename = "rotateFrame")]
    #[serde(default)]
    pub rotation_frames: Vec<RawRotationFrame>,
    #[serde(rename = "frame")]
    #[serde(default)]
    pub frames: Vec<RawGeneralFrame>
}

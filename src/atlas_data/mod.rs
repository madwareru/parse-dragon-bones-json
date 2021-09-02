use serde::Deserialize;
use std::collections::HashMap;

#[derive(Clone, Deserialize, Debug)]
struct RawAtlas {
    width: usize,
    height: usize,
    #[serde(rename = "SubTexture")]
    #[serde(default)]
    sub_textures: Vec<RawSubTexture>,
}

#[derive(Clone, Deserialize, Debug)]
struct RawSubTexture {
    name: String,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    #[serde(rename = "frameX")]
    #[serde(default)]
    frame_x: f32,
    #[serde(rename = "frameY")]
    #[serde(default)]
    frame_y: f32,
    #[serde(rename = "frameWidth")]
    #[serde(default)]
    frame_width: f32,
    #[serde(rename = "frameHeight")]
    #[serde(default)]
    frame_height: f32
}

#[derive(Clone, Debug)]
pub struct Atlas {
    pub width: usize,
    pub height: usize,
    pub sub_textures: HashMap<String, SubTexture>
}
impl From<&RawAtlas> for Atlas {
    fn from(source: &RawAtlas) -> Self {
        let mut sub_textures = HashMap::new();
        for sub in source.sub_textures.iter() {
            let name = sub.name.clone();
            let sub_texture = SubTexture::from(sub);
            sub_textures.insert(name, sub_texture);
        }
        Self {
            width: source.width,
            height: source.height,
            sub_textures
        }
    }
}

impl Atlas {
    pub fn parse(raw_bytes: &[u8]) -> Result<Self, serde_json::Error> {
        let raw: RawAtlas = serde_json::from_slice(raw_bytes)?;
        Ok((&raw).into())
    }
}


#[derive(Copy, Clone, Debug)]
pub struct SubTexture {
    pub rect: crate::shared_types::Rect,
    pub frame_rect: crate::shared_types::Rect
}

impl From<&RawSubTexture> for SubTexture {
    fn from(source: &RawSubTexture) -> Self {
        Self {
            rect: crate::shared_types::Rect {
                x: source.x,
                y: source.y,
                width: source.width,
                height: source.height
            },
            frame_rect: crate::shared_types::Rect {
                x: source.frame_x,
                y: source.frame_y,
                width: if source.frame_x == 0.0 && source.frame_width == 0.0 {
                    source.width
                } else {
                    source.frame_width
                },
                height: if source.frame_y == 0.0 && source.frame_height == 0.0 {
                    source.height
                } else {
                    source.frame_height
                }
            }
        }
    }
}

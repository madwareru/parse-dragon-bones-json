use serde::{Deserialize, Deserializer};
use crate::shared_types::{default_name, default_true};
use serde_json::{Value, Map, from_str};
use crate::skeleton_data::transform::RawTransform;

#[derive(Clone, Deserialize, Debug)]
pub struct RawSkinData {
    #[serde(rename = "slot")]
    #[serde(default)]
    pub slots: Vec<RawSkinSlot>,
}

#[derive(Clone, Deserialize, Debug)]
pub struct RawSkinSlot {
    #[serde(default = "default_name")]
    pub name: String,

    #[serde(rename = "display")]
    #[serde(default)]
    pub displays: Vec<RawDisplay>,
}

#[derive(Clone, Debug)]
pub enum RawBoundingBox {
    Vertices(Vec<f32>),
    Rectangle{ width: f32, height: f32 },
    Ellipse{ width: f32, height: f32 }
}

#[derive(Clone, Debug)]
pub enum RawDisplay {
    BoundingBoxDisplay {
        name: String,
        path: String,

        color: u32,

        sub_data: RawBoundingBox
    },
    Image {
        name: String,
        path: String,

        pivot: crate::shared_types::Point,

        transform: RawTransform,
    },
    Mesh {
        name: String,
        path: String,

        width: u32,

        height: u32,

        inherit_deform: bool,

        vertices: Vec<f32>,

        uvs: Vec<f32>,

        triangles: Vec<u32>,

        weights: Vec<f32>,

        slot_pose: Vec<f32>,

        bone_pose: Vec<f32>,

        edges: Vec<u32>,

        user_edges: Vec<u32>,
    },
}

#[derive(Deserialize)]
struct MeshDeserialized {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    path: String,

    width: u32,

    height: u32,

    #[serde(rename = "inheritDeform")]
    #[serde(default = "default_true")]
    inherit_deform: bool,

    #[serde(default)]
    vertices: Vec<f32>,

    #[serde(default)]
    uvs: Vec<f32>,

    #[serde(default)]
    triangles: Vec<u32>,

    #[serde(default)]
    weights: Vec<f32>,

    #[serde(rename = "slotPose")]
    #[serde(default)]
    slot_pose: Vec<f32>,

    #[serde(rename = "bonePose")]
    #[serde(default)]
    bone_pose: Vec<f32>,

    #[serde(default)]
    edges: Vec<u32>,

    #[serde(default)]
    user_edges: Vec<u32>,
}

impl<'de> Deserialize<'de> for RawDisplay {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        let value: serde_json::Value = Deserialize::deserialize(deserializer)?;
        let is_mesh = match &value {
            Value::Object(fields) => {
                if let Some(Value::String(tag)) = fields.get("type") {
                    Ok(tag.eq("mesh"))
                } else {
                    Ok(false)
                }
            },
            _ => Err(serde::de::Error::custom("Unexpected JSON field type! Object expected"))
        }?;

        if is_mesh {
            let serialized_string = serde_json::to_string(&value);
            let mesh_data: MeshDeserialized = match serialized_string {
                Ok(serialized_string) => {
                    match from_str(&serialized_string) {
                        Ok(m) => Ok(m),
                        Err(_) => Err(serde::de::Error::custom("Failed to deserialize mesh data"))
                    }
                },
                _ => Err(serde::de::Error::custom("Failed to serialize mesh data"))
            }?;
            return Ok(RawDisplay::Mesh {
                name: mesh_data.name,
                path: mesh_data.path,
                width: mesh_data.width,
                height: mesh_data.height,
                inherit_deform: mesh_data.inherit_deform,
                vertices: mesh_data.vertices,
                uvs: mesh_data.uvs,
                triangles: mesh_data.triangles,
                weights: mesh_data.weights,
                slot_pose: mesh_data.slot_pose,
                bone_pose: mesh_data.bone_pose,
                edges: mesh_data.edges,
                user_edges: mesh_data.user_edges
            });
        }

        match value {
            Value::Object(fields) => {
                if let Some(tag) = fields.get("type") {
                    match tag {
                        Value::String(s) if s.eq("boundingBox") => {
                            <RawDisplay>::parse_bounding_box::<D>(fields)
                        },
                        Value::String(s) if s.eq("image") => {
                            <RawDisplay>::parse_image::<D>(fields)
                        },
                        _ => Err(serde::de::Error::custom("Unexpected tag! Expected \"mesh\" or \"boundingBox\" or \"image\""))
                    }
                } else {
                    // Image could not contain a tag so in this case we are "defaulting" to Image
                    <RawDisplay>::parse_image::<D>(fields)
                }
            },
            _ => unreachable!()
        }
    }
}

impl RawDisplay {
    fn parse_image<'de, D: Deserializer<'de>>(fields: Map<String, Value>) -> Result<RawDisplay, D::Error> {
        let name = match fields.get("name") {
            None => Ok("".to_string()),
            Some(Value::String(s)) => Ok(s.clone()),
            _ => Err(serde::de::Error::custom("Unexpected JSON field type! String expected"))
        }?;

        let path = match fields.get("path") {
            None => Ok("".to_string()),
            Some(Value::String(s)) => Ok(s.clone()),
            _ => Err(serde::de::Error::custom("Unexpected JSON field type! String expected"))
        }?;

        let transform = match fields.get("transform") {
            None => Ok(RawTransform::default()),
            Some(Value::Object(obj)) => {
                let x = match obj.get("x") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                let y = match obj.get("y") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                let rotation = match obj.get("rotate") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                let skew = match obj.get("skew") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                let skew_x = match obj.get("skX") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                let skew_y = match obj.get("skY") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                let scale_x = match obj.get("scX") {
                    None => Ok(1.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                let scale_y = match obj.get("scY") {
                    None => Ok(1.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                Ok(RawTransform {
                    x,
                    y,
                    skew,
                    rotation,
                    skew_x,
                    skew_y,
                    scale_x,
                    scale_y
                })
            },
            _ => Err(serde::de::Error::custom("Unexpected JSON field type! Object expected"))
        }?;

        let pivot = match fields.get("pivot") {
            None => Ok(crate::shared_types::Point::default()),
            Some(Value::Object(obj)) => {
                let x = match obj.get("x") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                let y = match obj.get("y") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                Ok(crate::shared_types::Point { x, y })
            },
            _ => Err(serde::de::Error::custom("Unexpected JSON field type! Object expected"))
        }?;

        Ok(RawDisplay::Image {
            name,
            path,
            pivot,
            transform
        })
    }

    fn parse_bounding_box<'de, D: Deserializer<'de>>(fields: Map<String, Value>) -> Result<RawDisplay, D::Error> {
        let name = match fields.get("name") {
            None => Ok("".to_string()),
            Some(Value::String(s)) => Ok(s.clone()),
            _ => Err(serde::de::Error::custom("Unexpected JSON field type! String expected"))
        }?;

        let path = match fields.get("path") {
            None => Ok("".to_string()),
            Some(Value::String(s)) => Ok(s.clone()),
            _ => Err(serde::de::Error::custom("Unexpected JSON field type! String expected"))
        }?;

        let color = match fields.get("color") {
            None => Ok(default_black()),
            Some(Value::Number(n)) => Ok(n.as_u64().unwrap() as u32),
            _ => Err(serde::de::Error::custom("Unexpected JSON field type! String expected"))
        }?;

        let sub_data = match fields.get("subType") {
            Some(Value::String(s)) if s.eq("rectangle") => {
                let width = match fields.get("width") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                let height = match fields.get("height") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                Ok(RawBoundingBox::Rectangle { width, height })
            }
            None => {
                let width = match fields.get("width") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                let height = match fields.get("height") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                Ok(RawBoundingBox::Rectangle { width, height })
            }
            Some(Value::String(s)) if s.eq("ellipse") => {
                let width = match fields.get("width") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                let height = match fields.get("height") {
                    None => Ok(0.0),
                    Some(Value::Number(n)) => Ok(n.as_f64().unwrap() as f32),
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Number expected"))
                }?;
                Ok(RawBoundingBox::Ellipse { width, height })
            },
            Some(Value::String(s)) if s.eq("polygon") => {
                match fields.get("vertices") {
                    None => Ok(RawBoundingBox::Vertices(Default::default())),
                    Some(Value::Array(values)) => {
                        let mut verts = Vec::with_capacity(values.len());
                        let mut failed = false;
                        'conversion: for v in values.iter() {
                            match v {
                                Value::Number(n) => verts.push(n.as_f64().unwrap() as f32),
                                _ => {
                                    failed = true;
                                    break 'conversion;
                                }
                            }
                        }
                        if failed {
                            Err(serde::de::Error::custom("Unexpected JSON field type! Numerical array expected"))
                        } else {
                            Ok(RawBoundingBox::Vertices(verts))
                        }
                    },
                    _ => Err(serde::de::Error::custom("Unexpected JSON field type! Array expected"))
                }
            },
            _ => Err(serde::de::Error::custom("Unexpected JSON tag!"))
        }?;

        Ok(RawDisplay::BoundingBoxDisplay {
            name,
            path,
            color,
            sub_data
        })
    }
}

fn default_black() -> u32 {
    0x00000000
}

pub trait CastFrom<T> {
    fn cast(other: T) -> Self;
}
impl CastFrom<u32> for u16 {
    fn cast(other: u32) -> Self {
        other as u16
    }
}

#[derive(Clone, Debug)]
pub struct PurifiedMeshData<TIdx: CastFrom<u32>> {
    pub vertices: Vec<f32>,
    pub uvs: Vec<f32>,
    pub triangles: Vec<TIdx>,
    pub weights: Vec<f32>,
    pub edges: Vec<u32>,
    pub user_edges: Vec<u32>,
}

impl<TIdx: CastFrom<u32>> PurifiedMeshData<TIdx> {
    pub fn try_from(source: &RawDisplay, bone_count: usize) -> Option<Self> {
        match source {
            RawDisplay::BoundingBoxDisplay { .. } => None,
            RawDisplay::Image { .. } => None,
            RawDisplay::Mesh {
                vertices,
                uvs,
                triangles,
                weights: old_weights,
                edges,
                user_edges,
                ..
            } => {
                let vertices = vertices.clone();
                let triangles = triangles.iter().map(|it| TIdx::cast(*it)).collect::<Vec<TIdx>>();
                let uvs = uvs.clone();
                let edges = edges.clone();
                let user_edges = user_edges.clone();
                let mut weights = vec![0.0; vertices.len() * bone_count];
                let mut vertex_id = 0;
                let mut weights_offset = 0;
                loop {
                    if weights_offset >= old_weights.len() {
                        break;
                    }
                    let count = old_weights[weights_offset] as usize;
                    weights_offset += 1;
                    for _ in 0..count {
                        let bone_id = old_weights[weights_offset] as usize;
                        //let bone_id = bone_pose[bone_pose_id] as usize;
                        weights_offset += 1;
                        let weight = old_weights[weights_offset];
                        weights_offset += 1;
                        weights[vertex_id * bone_count + bone_id] = weight;
                    }
                    vertex_id += 1;
                }
                Some(Self {
                    vertices,
                    uvs,
                    triangles,
                    weights,
                    edges,
                    user_edges
                })
            }
        }
    }
}

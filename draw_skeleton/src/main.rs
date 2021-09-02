use macroquad::prelude::*;
use parse_dragon_bones_json::skeleton_data::RawSkeletonData;
use parse_dragon_bones_json::skeleton_data::armature::RawArmatureData;
use parse_dragon_bones_json::skeleton_data::transform::PurifiedTransform;
use parse_dragon_bones_json::skeleton_data::skin::{RawDisplay, PurifiedMeshData};
use std::collections::{HashMap, VecDeque};
use std::ops::{Mul, Add};
use parse_dragon_bones_json::skeleton_data::bone::RawBone;
use macroquad::miniquad::{TextureParams, TextureFormat, TextureWrap, Context};
use nalgebra::{Point3, Matrix3};
use nalgebra::LU;
use std::sync::Arc;
use std::borrow::Borrow;
use parse_dragon_bones_json::atlas_data::Atlas;

const COLORS: &[Color] = &[
    GOLD,
    ORANGE,
    PINK,
    RED,
    MAROON,
    GREEN,
    LIME,
    DARKGREEN,
    SKYBLUE,
    BLUE,
    DARKBLUE,
    PURPLE,
    VIOLET,
    DARKPURPLE,
    BEIGE,
    MAGENTA
];

pub struct BufferedDrawBatcher {
    vertex_buffer: Vec<Vertex>,
    index_buffer: Vec<u16>
}
impl BufferedDrawBatcher {
    pub fn new() -> Self {
        Self {
            vertex_buffer: Vec::new(),
            index_buffer: Vec::new()
        }
    }

    pub fn renderize_next_triangles(
        &mut self,
        vertices: impl Iterator<Item=Vertex>,
        indices: impl Iterator<Item=u16>,
        texture: Option<Texture2D>
    ) {
        self.vertex_buffer.clear();
        self.index_buffer.clear();
        self.vertex_buffer.extend(vertices);
        self.index_buffer.extend(indices);

        let mut quad_gl = unsafe {
            let InternalGlContext { quad_gl, .. } = get_internal_gl();
            quad_gl
        };

        quad_gl.texture(texture);
        quad_gl.draw_mode(DrawMode::Triangles);
        quad_gl.geometry(&self.vertex_buffer, &self.index_buffer);
    }
}

pub trait Drawable {
    fn draw(
        &self,
        draw_batcher: &mut BufferedDrawBatcher,
        diff_matrices: &[nalgebra::Matrix3<f32>],
        position_x: f32,
        position_y: f32,
        scale: f32
    );
}

#[derive(Clone)]
pub struct MeshDrawable {
    mesh_data: Arc<PurifiedMeshData<u16>>,
    texture: Texture2D
}
impl MeshDrawable {
    pub fn new(mesh_data: &PurifiedMeshData<u16>, texture: Texture2D) -> Self {
        Self {
            mesh_data: Arc::new(mesh_data.clone()),
            texture
        }
    }
}
impl Drawable for MeshDrawable {
    fn draw(
        &self,
        draw_batcher: &mut BufferedDrawBatcher,
        diff_matrices: &[nalgebra::Matrix3<f32>],
        position_x: f32,
        position_y: f32,
        scale: f32
    ) {
        let verts =
            buffer(&self.mesh_data.vertices, 2)
                .zip(buffer(&self.mesh_data.uvs, 2))
                .zip(buffer(&self.mesh_data.weights, diff_matrices.len()))
                .map(|((v, uv), w)| {
                    let pt: Point3<f32> = nalgebra::Point3::new(v[0], v[1], 1.0);
                    let vert_new = diff_matrices.iter()
                        .zip(w)
                        .filter(|(diff, &weight)| weight != 0.0)
                        .map(|(diff, &weight)| weight * (diff * pt))
                        .fold(
                            nalgebra::Point3::origin(),
                            |acc: Point3<f32>, x: Point3<f32>| [acc.x + x.x, acc.y + x.y, acc.z + x.z].into(),
                        );

                    Vertex::new(
                        position_x + vert_new.x * scale,
                        position_y + vert_new.y * scale,
                        0.0,
                        1.0 / 128.0 + uv[0] * 82.0 / 128.0,
                        1.0 / 128.0 + uv[1] * 114.0 / 128.0,
                        Color::new(1.0, 1.0, 1.0, 1.0),
                    )
                });

        let tris = self.mesh_data.triangles.iter().map(|it| *it);
        draw_batcher.renderize_next_triangles(verts, tris, Some(self.texture));
    }
}

#[derive(Copy, Clone)]
pub(crate) struct NamelessBone {
    id: usize,
    parent_id: Option<usize>,
    pub inherit_translation: bool,
    pub inherit_rotation: bool,
    pub inherit_scale: bool,
    pub inherit_reflection: bool,
    pub length: f32,
    pub transform: PurifiedTransform,
}

impl NamelessBone {
    fn from(bone_desc: (&RawBone, &[RawBone])) -> Self {
        let RawBone {
            name,
            parent,
            inherit_translation,
            inherit_rotation,
            inherit_scale,
            inherit_reflection,
            length,
            transform
        } = bone_desc.0;
        let id = (0..bone_desc.1.len()).find(|&id| bone_desc.1[id].name.eq(name)).unwrap();
        let parent_id = (0..bone_desc.1.len()).find(|&id| bone_desc.1[id].name.eq(parent));
        Self {
            id,
            parent_id,
            inherit_reflection: *inherit_reflection,
            inherit_scale: *inherit_scale,
            inherit_rotation: *inherit_rotation,
            inherit_translation: *inherit_translation,
            length: *length,
            transform: (*transform).into(),
        }
    }
}

#[macroquad::main("draw skeleton")]
async fn main() {
    let (mut quad_gl, mut ctx) = unsafe {
        let InternalGlContext { quad_gl, quad_context: ctx } = get_internal_gl();
        (quad_gl, ctx)
    };

    let texture_bytes = include_bytes!("../../src/test_assets/test_tex.png");
    let spider_texture_bytes = include_bytes!("../../src/test_assets/Spider_tex.png");
    let spider_texture_definition_bytes = include_bytes!("../../src/test_assets/Spider_tex.json");

    let texture = load_texture(ctx, texture_bytes);
    let spider_texture = load_texture(ctx, spider_texture_bytes);
    let spider_atlas = Atlas::parse(spider_texture_definition_bytes).unwrap();

    let skeleton_bytes = include_bytes!("../../src/test_assets/test_ske.json");
    let skeleton_data: RawSkeletonData = serde_json::from_slice(skeleton_bytes).unwrap();

    let armature = &skeleton_data.armatures[0];
    let mut mesh_drawables: Vec<Box<dyn Drawable>> = Vec::new();

    let mut bones = match armature {
        RawArmatureData::Armature { bones, skins, .. } => {
            let mut bone_vec = Vec::with_capacity(bones.len());
            for bone in bones.iter() {
                bone_vec.push(NamelessBone::from((bone, &bones[..])));
            }
            for skin in skins.iter() {
                for slot in skin.slots.iter() {
                    for display in slot.displays.iter() {
                        if let Some(mesh) = PurifiedMeshData::<u16>::try_from(display, bones.len()) {
                            mesh_drawables.push(Box::new(MeshDrawable::new(&mesh, texture)));
                        }
                    }
                }
            }
            bone_vec
        }
    };

    let mut bone_index = { // Topological sort
        let mut index = Vec::with_capacity(bones.len());
        let mut stack = VecDeque::new();
        for i in 0..bones.len() {
            let mut bone_ref = &bones[i];
            while bone_ref.parent_id.is_some() {
                bone_ref = match bone_ref.parent_id {
                    Some(id) if id == i => panic!("cycle detected!"),
                    Some(id) => {
                        stack.push_back(bone_ref.id);
                        &bones[id]
                    }
                    None => unreachable!()
                };
            }
            stack.push_back(bone_ref.id);
            'stack_unroll: loop {
                match stack.pop_back() {
                    None => { break 'stack_unroll; }
                    Some(id) => {
                        if !index.contains(&id) {
                            index.push(id)
                        }
                    }
                }
            }
        }
        index
    };

    let mut initial_matrices: Vec<nalgebra::Matrix3<f32>> = vec![nalgebra::Matrix3::identity(); bones.len()];
    for &bone_id in bone_index.iter() {
        let bone = &bones[bone_id];
        // Step 1: calculate matrix
        {
            let parent_transform = match bone.parent_id {
                None => nalgebra::Matrix3::identity(),
                Some(pid) => initial_matrices[pid]
            };
            let transition_local: nalgebra::Matrix3<f32> =
                nalgebra::Translation2::new(bone.transform.x, bone.transform.y).into();
            let rotation_matrix: nalgebra::Matrix3<f32> =
                nalgebra::Rotation2::new(bone.transform.rotation).into();
            let scale_matrix = nalgebra::Matrix3::new(
                bone.transform.scale_x, 0.0, 0.0,
                0.0, bone.transform.scale_y, 0.0,
                0.0, 0.0, 1.0,
            );
            initial_matrices[bone_id] = (parent_transform * transition_local * rotation_matrix * scale_matrix)
        }
    }
    for m in initial_matrices.iter_mut() {
        *m = m.try_inverse().unwrap();
    }

    let mut pose_matrices: Vec<nalgebra::Matrix3<f32>> = vec![nalgebra::Matrix3::identity(); bones.len()];
    let mut diff_matrices: Vec<nalgebra::Matrix3<f32>> = vec![nalgebra::Matrix3::identity(); bones.len()];
    let mut draw_buffer = BufferedDrawBatcher::new();

    loop {
        clear_background(Color::new(0.01, 0.0, 0.05, 1.0));

        draw_texture(spider_texture, 0.0, 0.0, WHITE);
        for (_, sub_tex) in spider_atlas.sub_textures.iter() {
            draw_rectangle_lines(
                sub_tex.rect.x,
                sub_tex.rect.y,
                sub_tex.rect.width,
                sub_tex.rect.height,
                1.0,
                WHITE
            );
            draw_rectangle_lines(
                sub_tex.rect.x + sub_tex.frame_rect.x,
                sub_tex.rect.y + sub_tex.frame_rect.y,
                sub_tex.frame_rect.width,
                sub_tex.frame_rect.height,
                1.0,
                GREEN
            );
        }

        let screen_center_x = screen_width() / 2.0;
        let screen_center_y = screen_height() / 2.0;

        bones[bone_index[4]].transform.rotation = (get_time() * 1.7).cos() as f32 * 0.1;

        bones[bone_index[1]].transform.rotation = std::f32::consts::PI / 2.0
            + (get_time() * 2.26).cos() as f32 * 0.15;
        bones[bone_index[2]].transform.rotation = std::f32::consts::PI / 2.0
            - (get_time() * 2.14).cos() as f32 * 0.14;

        for &bone_id in bone_index.iter() {
            let bone = &bones[bone_id];
            {
                let parent_transform = match bone.parent_id {
                    None => nalgebra::Matrix3::identity(),
                    Some(pid) => pose_matrices[pid]
                };
                let transition_local: nalgebra::Matrix3<f32> =
                    nalgebra::Translation2::new(bone.transform.x, bone.transform.y).into();
                let rotation_matrix: nalgebra::Matrix3<f32> =
                    nalgebra::Rotation2::new(bone.transform.rotation).into();
                let scale_matrix = nalgebra::Matrix3::new(
                    bone.transform.scale_x, 0.0, 0.0,
                    0.0, bone.transform.scale_y, 0.0,
                    0.0, 0.0, 1.0,
                );
                pose_matrices[bone_id] = (parent_transform * transition_local * rotation_matrix * scale_matrix);
                diff_matrices[bone_id] = pose_matrices[bone_id] * initial_matrices[bone_id];
            }
        }
        {
            for mesh_drawable in mesh_drawables.iter_mut() {
                mesh_drawable.draw(&mut draw_buffer, &diff_matrices, screen_center_x, screen_center_y, 2.0);
            }
        }

        for &bone_id in bone_index.iter() {
            let bone = &bones[bone_id];
            {
                let bone_color = COLORS[bone_id % COLORS.len()];
                let origin = pose_matrices[bone_id] * nalgebra::Point3::new(0.0, 0.0, 1.0);
                let dir = pose_matrices[bone_id] * nalgebra::Point3::new(bone.length, 0.0, 1.0);
                draw_line(
                    screen_center_x + origin.x * 2.0,
                    screen_center_y + origin.y * 2.0,
                    screen_center_x + dir.x * 2.0,
                    screen_center_y + dir.y * 2.0,
                    3.0,
                    bone_color,
                );
            }
        }
        next_frame().await;
    }
}

fn load_texture(mut ctx: &mut Context, texture_bytes: &[u8]) -> Texture2D {
    let img = image::load_from_memory(texture_bytes)
        .unwrap_or_else(|e| panic!("{}", e))
        .to_rgba8();
    let (img_width, img_height) = (img.width(), img.height());
    let raw_bytes = img.into_raw();

    Texture2D::from_miniquad_texture(
        miniquad::Texture::from_data_and_format(
            ctx,
            &raw_bytes[..],
            TextureParams {
                width: img_width,
                height: img_height,
                format: TextureFormat::RGBA8,
                filter: FilterMode::Nearest,
                wrap: TextureWrap::Clamp,
            },
        )
    )
}

fn buffer<T>(iterated: &[T], buffer_size: usize) -> impl Iterator<Item=&[T]> {
    let len = iterated.len();
    (0..len).step_by(buffer_size).map(
        move |id| &iterated[id..]
    )
}

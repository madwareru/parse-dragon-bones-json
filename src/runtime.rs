use macroquad::prelude::*;
use crate::skeleton_data::skin::{PurifiedMeshData, RawDisplay};
use std::sync::Arc;
use crate::atlas_data::{SubTexture, Atlas};
use crate::skeleton_data::transform::{PurifiedTransform, RawTransform};
use crate::skeleton_data::bone::RawBone;
use crate::skeleton_data::armature::RawArmatureData;
use std::collections::{VecDeque, HashMap};
use macroquad::miniquad::{TextureParams, TextureFormat, TextureWrap, Context};
use crate::skeleton_data::RawSkeletonData;
use std::ops::IndexMut;
use indextree::{Arena};
use crate::skeleton_data::ik::IkInfo;
use std::fmt::Debug;

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
    index_buffer: Vec<u16>,
}

impl BufferedDrawBatcher {
    pub fn new() -> Self {
        Self {
            vertex_buffer: Vec::new(),
            index_buffer: Vec::new(),
        }
    }

    pub fn renderize_next_triangles(
        &mut self,
        vertices: impl Iterator<Item=Vertex>,
        indices: impl Iterator<Item=u16>,
        texture: Option<Texture2D>,
    ) {
        self.vertex_buffer.clear();
        self.index_buffer.clear();
        self.vertex_buffer.extend(vertices);
        self.index_buffer.extend(indices);

        let quad_gl = unsafe {
            let InternalGlContext { quad_gl, .. } = get_internal_gl();
            quad_gl
        };

        quad_gl.texture(texture);
        quad_gl.draw_mode(DrawMode::Triangles);
        quad_gl.geometry(&self.vertex_buffer, &self.index_buffer);
    }
}

pub trait Drawable {
    fn get_draw_order(&self) -> i32;
    fn draw(
        &self,
        draw_batcher: &mut BufferedDrawBatcher,
        pose_matrices: &[nalgebra::Matrix3<f32>],
        diff_matrices: &[nalgebra::Matrix3<f32>],
        position_x: f32,
        position_y: f32,
        scale: f32,
    );
    fn instantiate(&self) -> Box<dyn Drawable>;
}

#[derive(Clone)]
pub struct MeshDrawable {
    mesh_data: Arc<PurifiedMeshData<u16>>,
    texture: Texture2D,
    atlas_size: [f32; 2],
    atlas_sub_texture: SubTexture,
    draw_order: i32,
}

impl MeshDrawable {
    pub fn new(
        mesh_data: &PurifiedMeshData<u16>,
        texture: Texture2D,
        atlas_size: [f32; 2],
        atlas_sub_texture: SubTexture,
        draw_order: i32,
    ) -> Self {
        Self {
            mesh_data: Arc::new(mesh_data.clone()),
            texture,
            atlas_size,
            atlas_sub_texture,
            draw_order,
        }
    }
}

impl Drawable for MeshDrawable {
    fn get_draw_order(&self) -> i32 { self.draw_order }

    fn draw(
        &self,
        draw_batcher: &mut BufferedDrawBatcher,
        _pose_matrices: &[nalgebra::Matrix3<f32>],
        diff_matrices: &[nalgebra::Matrix3<f32>],
        position_x: f32,
        position_y: f32,
        scale: f32,
    ) {
        let verts =
            buffer(&self.mesh_data.vertices, 2)
                .zip(buffer(&self.mesh_data.uvs, 2))
                .zip(buffer(&self.mesh_data.weights, diff_matrices.len()))
                .map(|((v, uv), w)| {
                    let pt: nalgebra::Point3<f32> = nalgebra::Point3::new(v[0], v[1], 1.0);
                    let vert_new = diff_matrices.iter()
                        .zip(w)
                        .filter(|(_, &weight)| weight != 0.0)
                        .map(|(diff, &weight)| weight * (diff * pt))
                        .fold(
                            nalgebra::Point3::origin(),
                            |acc: nalgebra::Point3<f32>, x: nalgebra::Point3<f32>| [acc.x + x.x, acc.y + x.y, acc.z + x.z].into(),
                        );

                    Vertex::new(
                        position_x + vert_new.x * scale,
                        position_y + vert_new.y * scale,
                        0.0,
                        self.atlas_sub_texture.rect.x / self.atlas_size[0]
                            + uv[0] * self.atlas_sub_texture.rect.width / self.atlas_size[0],
                        self.atlas_sub_texture.rect.y / self.atlas_size[1]
                            + uv[1] * self.atlas_sub_texture.rect.height / self.atlas_size[1],
                        Color::new(1.0, 1.0, 1.0, 1.0),
                    )
                });

        let tris = self.mesh_data.triangles.iter().map(|it| *it);
        draw_batcher.renderize_next_triangles(verts, tris, Some(self.texture));
    }

    fn instantiate(&self) -> Box<dyn Drawable> {
        let mesh_data = self.mesh_data.clone();
        let texture = self.texture;
        let atlas_size = self.atlas_size;
        let atlas_sub_texture = self.atlas_sub_texture.clone();
        let draw_order = self.draw_order;
        Box::new(Self {
            mesh_data,
            texture,
            atlas_size,
            atlas_sub_texture,
            draw_order,
        })
    }
}

#[derive(Clone)]
pub struct ImageDrawable {
    texture: Texture2D,
    atlas_size: [f32; 2],
    atlas_sub_texture: SubTexture,
    parent_bone_id: usize,
    transform: PurifiedTransform,
    vertices: [nalgebra::Point3<f32>; 4],
    uvs: [(f32, f32); 4],
    draw_order: i32,
}

impl ImageDrawable {
    pub fn new(
        texture: Texture2D,
        atlas_size: [f32; 2],
        atlas_sub_texture: SubTexture,
        parent_bone_id: usize,
        transform: RawTransform,
        pivot: crate::shared_types::Point,
        draw_order: i32,
    ) -> Self {
        let left = -pivot.x * atlas_sub_texture.frame_rect.width;
        let top = -pivot.y * atlas_sub_texture.frame_rect.height;

        let x_alpha_0 = (-atlas_sub_texture.frame_rect.x)
            / atlas_sub_texture.frame_rect.width;

        let x_alpha_1 = (atlas_sub_texture.rect.width - atlas_sub_texture.frame_rect.x)
            / atlas_sub_texture.frame_rect.width;

        let y_alpha_0 = (-atlas_sub_texture.frame_rect.y)
            / atlas_sub_texture.frame_rect.height;

        let y_alpha_1 = (atlas_sub_texture.rect.height - atlas_sub_texture.frame_rect.y)
            / atlas_sub_texture.frame_rect.height;

        let vertices = [
            nalgebra::Point3::from(
                [
                    left + x_alpha_0 * atlas_sub_texture.frame_rect.width,
                    top + y_alpha_0 * atlas_sub_texture.frame_rect.height,
                    1.0
                ]
            ),
            nalgebra::Point3::from(
                [
                    left + x_alpha_1 * atlas_sub_texture.frame_rect.width,
                    top + y_alpha_0 * atlas_sub_texture.frame_rect.height,
                    1.0
                ]
            ),
            nalgebra::Point3::from(
                [
                    left + x_alpha_0 * atlas_sub_texture.frame_rect.width,
                    top + y_alpha_1 * atlas_sub_texture.frame_rect.height,
                    1.0
                ]
            ),
            nalgebra::Point3::from(
                [
                    left + x_alpha_1 * atlas_sub_texture.frame_rect.width,
                    top + y_alpha_1 * atlas_sub_texture.frame_rect.height,
                    1.0
                ]
            )
        ];

        let uvs = [
            (
                atlas_sub_texture.rect.x / atlas_size[0],
                atlas_sub_texture.rect.y / atlas_size[1]
            ),
            (
                (atlas_sub_texture.rect.x + atlas_sub_texture.rect.width) / atlas_size[0],
                atlas_sub_texture.rect.y / atlas_size[1]
            ),
            (
                atlas_sub_texture.rect.x / atlas_size[0],
                (atlas_sub_texture.rect.y + atlas_sub_texture.rect.height) / atlas_size[1]
            ),
            (
                (atlas_sub_texture.rect.x + atlas_sub_texture.rect.width) / atlas_size[0],
                (atlas_sub_texture.rect.y + atlas_sub_texture.rect.height) / atlas_size[1]
            )
        ];
        Self {
            texture,
            atlas_size,
            atlas_sub_texture,
            parent_bone_id,
            transform: transform.into(),
            vertices,
            uvs,
            draw_order,
        }
    }
}

impl Drawable for ImageDrawable {
    fn get_draw_order(&self) -> i32 { self.draw_order }

    fn draw(
        &self,
        draw_batcher: &mut BufferedDrawBatcher,
        pose_matrices: &[nalgebra::Matrix3<f32>],
        _diff_matrices: &[nalgebra::Matrix3<f32>],
        position_x: f32,
        position_y: f32,
        scale: f32,
    ) {
        let transition_local: nalgebra::Matrix3<f32> =
            nalgebra::Translation2::new(self.transform.x, self.transform.y).into();
        let rotation_matrix: nalgebra::Matrix3<f32> =
            nalgebra::Rotation2::new(self.transform.rotation).into();
        let scale_matrix = nalgebra::Matrix3::new(
            self.transform.scale_x, 0.0, 0.0,
            0.0, self.transform.scale_y, 0.0,
            0.0, 0.0, 1.0,
        );
        let mat = pose_matrices[self.parent_bone_id] * transition_local * rotation_matrix * scale_matrix;
        let indices = [0, 1, 2, 1, 2, 3].iter().map(|it| *it as u16);
        let verts =
            self.vertices.iter()
                .zip(self.uvs.iter())
                .map(|(&v, uv)| {
                    let vert_new: nalgebra::Point3<f32> = mat * v;

                    Vertex::new(
                        position_x + vert_new.x * scale,
                        position_y + vert_new.y * scale,
                        0.0,
                        uv.0,
                        uv.1,
                        Color::new(1.0, 1.0, 1.0, 1.0),
                    )
                });
        draw_batcher.renderize_next_triangles(verts, indices, Some(self.texture));
    }

    fn instantiate(&self) -> Box<dyn Drawable> {
        let texture = self.texture;
        let atlas_size = self.atlas_size;
        let atlas_sub_texture = self.atlas_sub_texture.clone();
        let draw_order = self.draw_order;
        Box::new(Self {
            texture,
            atlas_size,
            atlas_sub_texture,
            parent_bone_id: self.parent_bone_id,
            transform: self.transform,
            vertices: self.vertices.clone(),
            uvs: self.uvs,
            draw_order,
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NamelessBone {
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

fn buffer<T>(iterated: &[T], buffer_size: usize) -> impl Iterator<Item=&[T]> {
    let len = iterated.len();
    (0..len).step_by(buffer_size).map(
        move |id| &iterated[id..]
    )
}

#[derive(Copy, Clone, Debug)]
struct BoneInfo {
    id: usize,
    is_dirty: bool,
}

pub struct RuntimeArmature {
    ticked_time: f32,
    ms_per_tick: f32,
    ik: Arc<Vec<IkInfo>>,
    bone_tree: Arena<BoneInfo>,
    tree_handles: Vec<indextree::NodeId>,
    bone_lookup: Arc<HashMap<String, usize>>,
    pose_bones: Arc<Vec<NamelessBone>>,
    bones: Vec<NamelessBone>,
    drawables: Vec<Box<dyn Drawable>>,
    initial_matrices: Arc<Vec<nalgebra::Matrix3<f32>>>,
    pose_matrices: Vec<nalgebra::Matrix3<f32>>,
    diff_matrices: Vec<nalgebra::Matrix3<f32>>,
    buffer_deque: VecDeque<indextree::NodeId>,
    animations: Arc<Vec<AnimationData>>,
    start_animation_id: usize,
    current_animation_id: usize,
    current_animation_ticks: usize
}

impl RuntimeArmature {
    pub fn get_bone_by_name(&self, bone_name: &str) -> Option<usize> {
        self.bone_lookup.get(bone_name).map(|&it| it)
    }
}

impl RuntimeArmature {
    pub fn instantiate(&self) -> Self {
        let bones = self.bones.clone();
        let bone_lookup = self.bone_lookup.clone();
        let initial_matrices = self.initial_matrices.clone();
        let pose_matrices = self.pose_matrices.clone();
        let diff_matrices = self.diff_matrices.clone();
        let drawables = self.drawables.iter().map(|it| it.instantiate()).collect();
        let ms_per_tick = self.ms_per_tick;
        Self {
            ticked_time: 0.0,
            ms_per_tick,
            bones,
            pose_bones: self.pose_bones.clone(),
            drawables,
            initial_matrices,
            pose_matrices,
            diff_matrices,
            bone_lookup,
            bone_tree: self.bone_tree.clone(),
            tree_handles: self.tree_handles.clone(),
            buffer_deque: VecDeque::new(),
            ik: self.ik.clone(),
            animations: self.animations.clone(),
            start_animation_id: self.start_animation_id,
            current_animation_id: self.start_animation_id,
            current_animation_ticks: 0
        }
    }

    pub fn extract(
        raw_armature: &RawArmatureData,
        atlas: &crate::atlas_data::Atlas,
        texture: Texture2D,
        frame_rate: u32,
    ) -> (String, Self) {
        match raw_armature {
            RawArmatureData::Armature { name, bones, skins, slots, ik, animations, default_actions, .. } => {
                let ik = Arc::new(ik.clone());
                let mut drawables: Vec<Box<dyn Drawable>> = Vec::new();
                let mut bone_vec = Vec::with_capacity(bones.len());
                let mut bone_lookup = HashMap::new();
                for bone in bones.iter() {
                    bone_lookup.insert(bone.name.clone(), bone_vec.len());
                    bone_vec.push(NamelessBone::from((bone, &bones[..])));
                }
                for skin in skins.iter() {
                    for slot in skin.slots.iter() {
                        let (parent_bone_name, draw_order) =
                            slots.iter().enumerate().find_map(|(id, it)| {
                                if it.name.eq(&slot.name) { Some((&it.parent as &str, id as i32)) } else { None }
                            }).unwrap_or(("", i32::MIN));
                        let parent_bone_id = (0..bones.len()).find(|&id| bones[id].name.eq(parent_bone_name));
                        for display in slot.displays.iter() {
                            let sub_texture = display.get_rect(atlas);
                            if let Some(mesh) = PurifiedMeshData::<u16>::try_from(display, bones.len()) {
                                drawables.push(Box::new(
                                    MeshDrawable::new(
                                        &mesh,
                                        texture,
                                        [atlas.width as f32, atlas.height as f32],
                                        sub_texture.unwrap(),
                                        draw_order,
                                    )
                                ));
                            }
                            if let &RawDisplay::Image { pivot, transform, .. } = display {
                                drawables.push(Box::new(
                                    ImageDrawable::new(
                                        texture,
                                        [atlas.width as f32, atlas.height as f32],
                                        sub_texture.unwrap(),
                                        parent_bone_id.unwrap(),
                                        transform,
                                        pivot,
                                        draw_order,
                                    )
                                ));
                            }
                        }
                    }
                }
                let mut initial_matrices: Vec<nalgebra::Matrix3<f32>> = vec![nalgebra::Matrix3::identity(); bones.len()];
                for bone_id in 0..bone_vec.len() {
                    let bone = &bone_vec[bone_id];
                    let flattened_rotation = if bone.inherit_rotation {
                        bone.transform.rotation
                    } else {
                        let mut bone = bone;
                        let mut rot = bone.transform.rotation;
                        while let Some(pid) = bone.parent_id {
                            bone = &bone_vec[pid];
                            rot -= bone.transform.rotation;
                        }
                        rot
                    };
                    let flattened_scale = if bone.inherit_scale {
                        (bone.transform.scale_x, bone.transform.scale_y)
                    } else {
                        let mut bone = bone;
                        let mut scale = (bone.transform.scale_x, bone.transform.scale_y);
                        while let Some(pid) = bone.parent_id {
                            bone = &bone_vec[pid];
                            scale.0 -= bone.transform.scale_x;
                            scale.1 -= bone.transform.scale_y;
                        }
                        scale
                    };
                    let flattened_translation = (bone.transform.x, bone.transform.y);
                    // Step 1: calculate matrix
                    {
                        let parent_transform = match bone.parent_id {
                            None => nalgebra::Matrix3::identity(),
                            Some(pid) => initial_matrices[pid]
                        };
                        let transition_local: nalgebra::Matrix3<f32> =
                            nalgebra::Translation2::new(flattened_translation.0, flattened_translation.1).into();
                        let rotation_matrix: nalgebra::Matrix3<f32> =
                            nalgebra::Rotation2::new(flattened_rotation).into();
                        let scale_matrix = nalgebra::Matrix3::new(
                            flattened_scale.0, 0.0, 0.0,
                            0.0, flattened_scale.1, 0.0,
                            0.0, 0.0, 1.0,
                        );
                        initial_matrices[bone_id] = parent_transform * transition_local * rotation_matrix * scale_matrix;
                    }
                }
                for m in initial_matrices.iter_mut() {
                    *m = m.try_inverse().unwrap();
                }

                let pose_matrices = initial_matrices.clone();
                let diff_matrices: Vec<nalgebra::Matrix3<f32>> = vec![nalgebra::Matrix3::identity(); bone_vec.len()];

                let mut bone_tree = Arena::new();
                let mut tree_handles: Vec<indextree::NodeId> = Vec::with_capacity(bone_vec.len());
                for i in 0..bone_vec.len() {
                    let bone = &bone_vec[i];
                    let handle = bone_tree.new_node(BoneInfo { id: i, is_dirty: true });
                    if let Some(pid) = bone.parent_id {
                        let parent_handle = tree_handles[pid];
                        parent_handle.append(handle, &mut bone_tree);
                    }
                    tree_handles.push(handle);
                }

                let mut animations_vec: Vec<AnimationData> = Vec::new();
                let mut bezier_buffer: Vec<CubicBezierRegion> = Vec::new();
                for anim in animations.iter() {
                    let name = anim.name.clone();
                    let duration_in_ticks = anim.duration as usize;
                    let play_times = anim.play_times as usize;
                    let mut animation_data = AnimationData {
                        name,
                        duration_in_ticks,
                        play_times,
                        rotation_tracks: Vec::new(),
                        transition_tracks: Vec::new(),
                        scaling_tracks: Vec::new()
                    };
                    for bone_timeline in anim.bone_timelines.iter() {
                        let bone_id = *bone_lookup.get(&bone_timeline.bone_name).unwrap();
                        if bone_timeline.frames.len() > 0 {
                            //todo: support sometime
                        } else {
                            if bone_timeline.rotation_frames.len() > 0 {
                                let mut tick_now: usize = 0;
                                let mut animation_track = AnimationTrack {
                                    bone_id,
                                    regions: Vec::new()
                                };
                                for i in 0..bone_timeline.rotation_frames.len()-1 {
                                    let frame = &bone_timeline.rotation_frames[i];
                                    let next_frame = &bone_timeline.rotation_frames[i+1];
                                    let rotation = bone_vec[bone_id].transform.rotation
                                        + frame.rotate.to_radians()
                                        * if frame.clockwise == 1 { -1.0 } else { 1.0 };
                                    let next_rotation = bone_vec[bone_id].transform.rotation
                                        + next_frame.rotate.to_radians()
                                        * if next_frame.clockwise == 1 { -1.0 } else { 1.0 };

                                    animation_track.regions.push(
                                        SamplingRegion {
                                            start_sample: RotationSample{ theta: rotation },
                                            end_sample: RotationSample{ theta: next_rotation },
                                            start_tick: tick_now,
                                            end_tick: tick_now + frame.duration as usize,
                                            tween_easing: TweenEasing::parse(
                                                &mut bezier_buffer,
                                                frame.tween_easing,
                                                &frame.curve
                                            )
                                        }
                                    );
                                    tick_now += bone_timeline.rotation_frames[i].duration as usize;
                                }
                                animation_data.rotation_tracks.push(animation_track);
                            }
                            if bone_timeline.translate_frames.len() > 0 {
                                let mut tick_now: usize = 0;
                                let mut animation_track = AnimationTrack {
                                    bone_id,
                                    regions: Vec::new()
                                };
                                for i in 0..bone_timeline.translate_frames.len()-1 {
                                    let frame = &bone_timeline.translate_frames[i];
                                    let next_frame = &bone_timeline.translate_frames[i+1];

                                    let translation_x = bone_vec[bone_id].transform.x + frame.x;
                                    let translation_y = bone_vec[bone_id].transform.y + frame.y;

                                    let next_translation_x = bone_vec[bone_id].transform.x + next_frame.x;
                                    let next_translation_y = bone_vec[bone_id].transform.y + next_frame.y;

                                    animation_track.regions.push(
                                        SamplingRegion {
                                            start_sample: TransitionSample{
                                                x: translation_x,
                                                y: translation_y
                                            },
                                            end_sample: TransitionSample{
                                                x: next_translation_x,
                                                y: next_translation_y
                                            },
                                            start_tick: tick_now,
                                            end_tick: tick_now + frame.duration as usize,
                                            tween_easing: TweenEasing::parse(
                                                &mut bezier_buffer,
                                                frame.tween_easing,
                                                &frame.curve
                                            )
                                        }
                                    );
                                    tick_now += bone_timeline.translate_frames[i].duration as usize;
                                }
                                animation_data.transition_tracks.push(animation_track);
                            }
                            if bone_timeline.scale_frames.len() > 0 {
                                let mut tick_now: usize = 0;
                                let mut animation_track = AnimationTrack {
                                    bone_id,
                                    regions: Vec::new()
                                };
                                for i in 0..bone_timeline.scale_frames.len()-1 {
                                    let frame = &bone_timeline.scale_frames[i];
                                    let next_frame = &bone_timeline.scale_frames[i+1];

                                    let scale_x = frame.x;
                                    let scale_y = frame.y;

                                    let next_scale_x = next_frame.x;
                                    let next_scale_y = next_frame.y;

                                    animation_track.regions.push(
                                        SamplingRegion {
                                            start_sample: ScalingSample{
                                                scale_x,
                                                scale_y
                                            },
                                            end_sample: ScalingSample{
                                                scale_x: next_scale_x,
                                                scale_y: next_scale_y
                                            },
                                            start_tick: tick_now,
                                            end_tick: tick_now + frame.duration as usize,
                                            tween_easing: TweenEasing::parse(
                                                &mut bezier_buffer,
                                                frame.tween_easing,
                                                &frame.curve
                                            )
                                        }
                                    );
                                    tick_now += bone_timeline.scale_frames[i].duration as usize;
                                }
                                animation_data.scaling_tracks.push(animation_track);
                            }
                        }
                    }
                    animations_vec.push(animation_data);
                }

                let start_animation_id =
                    default_actions.iter()
                        .find(|it| it.action_type.eq("play"))
                        .and_then(|it| {
                            (0..animations_vec.len()).find(|&id| animations_vec[id].name.eq(&it.goto_and_play))
                        })
                        .unwrap_or(0);

                (
                    name.clone(),
                    Self {
                        ticked_time: 0.0,
                        ms_per_tick: 1.0 / frame_rate as f32,
                        pose_bones: Arc::new(bone_vec.clone()),
                        bones: bone_vec,
                        drawables,
                        initial_matrices: Arc::new(initial_matrices),
                        pose_matrices,
                        diff_matrices,
                        bone_lookup: Arc::new(bone_lookup),
                        bone_tree,
                        tree_handles,
                        buffer_deque: VecDeque::new(),
                        ik,
                        animations: Arc::new(animations_vec),
                        start_animation_id,
                        current_animation_id: start_animation_id,
                        current_animation_ticks: 0
                    }
                )
            }
        }
    }

    pub fn goto_and_play(&mut self, animation_name: &str) {
        let animation_id = (0..self.animations.len()).find(|&id| {
            self.animations[id].name.eq(animation_name)
        });
        if let Some(animation_id) = animation_id {
            self.current_animation_id = animation_id;
            self.current_animation_ticks = 0;
            for i in 0..self.bones.len() {
                self.bones[i].transform = self.pose_bones[i].transform;
            }
            for &node_id in self.tree_handles.iter() {
                let mut node = self.bone_tree.get_mut(node_id).unwrap().get_mut();
                node.is_dirty = true;
            }
        }
    }

    pub fn update_animation(&mut self, dt: f32) {
        self.ticked_time += dt;
        while self.ticked_time >= self.ms_per_tick {
            self.ticked_time -= self.ms_per_tick;
            self.tick_animation();
        }
        self.update_matrices();
        self.update_ik();
        self.update_matrices();
    }

    pub fn update_animation_ex(
        &mut self,
        dt: f32,
        post_process_animation: impl FnOnce(&mut BonesMut) -> (),
    ) {
        self.ticked_time += dt;
        while self.ticked_time >= self.ms_per_tick {
            self.ticked_time -= self.ms_per_tick;
            self.tick_animation();
        }
        self.update_matrices();
        {
            let mut bones_mut = BonesMut { armature: self };
            post_process_animation(&mut bones_mut);
        }
        self.update_matrices();
        self.update_ik();
        self.update_matrices();
    }

    fn tick_animation(&mut self) {
        let play_times = self.animations[self.current_animation_id].play_times;
        let duration_in_ticks = self.animations[self.current_animation_id].duration_in_ticks;
        if play_times != 0 && duration_in_ticks * play_times <= self.current_animation_ticks {
            return;
        }
        if play_times == 0 && duration_in_ticks <= self.current_animation_ticks {
            self.current_animation_ticks = self.current_animation_ticks % duration_in_ticks;
        }

        let num_rotation_tracks = self.animations[self.current_animation_id].rotation_tracks.len();
        let num_transition_tracks = self.animations[self.current_animation_id].transition_tracks.len();
        let num_scaling_tracks = self.animations[self.current_animation_id].scaling_tracks.len();
        for i in 0..num_rotation_tracks {
            let (bone_id, current_frame) = {
                let track = &self.animations[self.current_animation_id].rotation_tracks[i];
                (
                    track.bone_id,
                    track.regions.iter().find(|&it| {
                        it.start_tick <= self.current_animation_ticks &&
                        it.end_tick >= self.current_animation_ticks
                    })
                )
            };
            if let Some(current_frame) = current_frame {
                let interpolated_rotation = current_frame.interpolate(self.current_animation_ticks);
                let mut bones_mut = BonesMut { armature: self };
                bones_mut[bone_id].transform.rotation = interpolated_rotation.theta;
            }
        }
        for i in 0..num_transition_tracks {
            let (bone_id, current_frame) = {
                let track = &self.animations[self.current_animation_id].transition_tracks[i];
                (
                    track.bone_id,
                    track.regions.iter().find(|&it| {
                        it.start_tick <= self.current_animation_ticks &&
                            it.end_tick >= self.current_animation_ticks
                    })
                )
            };
            if let Some(current_frame) = current_frame {
                let interpolated_translation = current_frame.interpolate(self.current_animation_ticks);
                let mut bones_mut = BonesMut { armature: self };
                bones_mut[bone_id].transform.x = interpolated_translation.x;
                bones_mut[bone_id].transform.y = interpolated_translation.y;
            }
        }
        for i in 0..num_scaling_tracks {
            let (bone_id, current_frame) = {
                let track = &self.animations[self.current_animation_id].scaling_tracks[i];
                (
                    track.bone_id,
                    track.regions.iter().find(|&it| {
                        it.start_tick <= self.current_animation_ticks &&
                            it.end_tick >= self.current_animation_ticks
                    })
                )
            };
            if let Some(current_frame) = current_frame {
                let interpolated_scale = current_frame.interpolate(self.current_animation_ticks);
                let mut bones_mut = BonesMut { armature: self };
                bones_mut[bone_id].transform.scale_x = interpolated_scale.scale_x;
                bones_mut[bone_id].transform.scale_y = interpolated_scale.scale_y;
            }
        }
        self.current_animation_ticks += 1;
    }

    fn update_ik(&mut self) {
        for ik_info_id in 0..self.ik.len() {
            let (bone_id, effector_bone_id, chain_length, bend_positive) = {
                let bone_id = self.get_bone_by_name(&self.ik[ik_info_id].bone).unwrap();
                let effector_bone_id = self.get_bone_by_name(&self.ik[ik_info_id].target).unwrap();
                (bone_id, effector_bone_id, self.ik[ik_info_id].chain_length, self.ik[ik_info_id].bend_positive)
            };

            let effector_position: nalgebra::Point3<f32> =
                self.pose_matrices[effector_bone_id] *
                    nalgebra::Point3::new(0.0, 0.0, 1.0);

            match chain_length {
                0 => {
                    let mut bone = &self.bones[bone_id];
                    let origin: nalgebra::Point3<f32> = self.pose_matrices[bone_id] * nalgebra::Point3::new(0.0, 0.0, 1.0);
                    let delta = effector_position - origin;
                    let mut rotation = delta.y.atan2(delta.x);
                    if bone.inherit_rotation {
                        while let Some(pid) = bone.parent_id {
                            bone = &self.bones[pid];
                            rotation -= bone.transform.rotation;
                        }
                    }
                    let mut bones_mut = BonesMut { armature: self };
                    bones_mut[bone_id].transform.rotation = rotation;
                }
                1 => {
                    let upper_bone_id = self.bones[bone_id].parent_id.unwrap();

                    let bone_lower = &self.bones[bone_id];
                    let bone_upper = &self.bones[upper_bone_id];

                    let l1: f32 = bone_lower.length;
                    let l2: f32 = bone_upper.length;

                    let origin: nalgebra::Point3<f32> = self.pose_matrices[upper_bone_id] *
                        nalgebra::Point3::new(0.0, 0.0, 1.0);

                    let delta = effector_position.clone() - origin.clone();
                    let direction = delta.normalize();

                    let mut angle_decrement = 0.0;
                    let mut bone = bone_upper;
                    if bone.inherit_rotation {
                        while let Some(pid) = bone.parent_id {
                            bone = &self.bones[pid];
                            angle_decrement += bone.transform.rotation;
                        }
                    }

                    let (lower_rotation, upper_rotation) = if delta.magnitude() > l1 + l2 {
                        let upper_rotation = delta.y.atan2(delta.x) - angle_decrement;
                        (0.0, upper_rotation)
                    } else {
                        let k2: f32 = l1 * l1 - l2 * l2;
                        let k1 = delta.magnitude();

                        let d = (k1 * k1 - k2) / (2.0 * k1);
                        let a = (d / l2).acos();

                        let mat: nalgebra::Matrix3<f32> = if bend_positive {
                            nalgebra::Rotation2::new(-a).into()
                        } else {
                            nalgebra::Rotation2::new(a).into()
                        };
                        let direction: nalgebra::Point3<f32> = nalgebra::Point3::new(
                            direction.x,
                            direction.y,
                            1.0,
                        );
                        let delta = mat * direction;
                        let knee_position: nalgebra::Point3<f32> =
                            nalgebra::Point3::new(
                                origin.x + delta.x * l2,
                                origin.y + delta.y * l2,
                                1.0
                            );

                        let upper_rotation = delta.y.atan2(delta.x) - angle_decrement;
                        let lower_rotation =
                            (effector_position.y - knee_position.y).atan2(effector_position.x - knee_position.x)
                            - upper_rotation - angle_decrement;
                        (lower_rotation, upper_rotation)
                    };
                    let mut bones_mut = BonesMut { armature: self };
                    bones_mut[bone_id].transform.rotation = lower_rotation;
                    bones_mut[upper_bone_id].transform.rotation = upper_rotation;
                }
                _ => panic!("unsupported ik chain length!")
            }
        }
    }

    fn update_matrices(&mut self) {
        for &node_id in self.tree_handles.iter() {
            let mut node = self.bone_tree.get_mut(node_id).unwrap().get_mut();
            if node.is_dirty {
                node.is_dirty = false;
            } else {
                continue;
            }
            let bone_id = node.id;
            let bone = &self.bones[bone_id];
            let flattened_rotation = if bone.inherit_rotation {
                bone.transform.rotation
            } else {
                let mut bone = bone;
                let mut rot = bone.transform.rotation;
                while let Some(pid) = bone.parent_id {
                    bone = &self.bones[pid];
                    rot -= bone.transform.rotation;
                }
                rot
            };
            let flattened_scale = if bone.inherit_scale {
                (bone.transform.scale_x, bone.transform.scale_y)
            } else {
                let mut bone = bone;
                let mut scale = (bone.transform.scale_x, bone.transform.scale_y);
                while let Some(pid) = bone.parent_id {
                    bone = &self.bones[pid];
                    scale.0 /= bone.transform.scale_x;
                    scale.1 /= bone.transform.scale_y;
                }
                scale
            };
            let flattened_translation = (bone.transform.x, bone.transform.y);
            {
                let parent_transform = match bone.parent_id {
                    None => nalgebra::Matrix3::identity(),
                    Some(pid) => self.pose_matrices[pid]
                };
                let transition_local: nalgebra::Matrix3<f32> =
                    nalgebra::Translation2::new(flattened_translation.0, flattened_translation.1).into();
                let rotation_matrix: nalgebra::Matrix3<f32> =
                    nalgebra::Rotation2::new(flattened_rotation).into();
                let scale_matrix = nalgebra::Matrix3::new(
                    flattened_scale.0, 0.0, 0.0,
                    0.0, flattened_scale.1, 0.0,
                    0.0, 0.0, 1.0,
                );
                self.pose_matrices[bone_id] = parent_transform * transition_local * rotation_matrix * scale_matrix;
                self.diff_matrices[bone_id] = self.pose_matrices[bone_id] * self.initial_matrices[bone_id];
            }
        }
    }

    pub fn draw(&mut self, draw_batch: &mut BufferedDrawBatcher, position_x: f32, position_y: f32, scale: f32) {
        self.drawables.sort_by(|lhs, rhs| lhs.get_draw_order().cmp(&rhs.get_draw_order()));
        for drawable in self.drawables.iter() {
            drawable.draw(
                draw_batch,
                &self.pose_matrices,
                &self.diff_matrices,
                position_x,
                position_y,
                scale,
            );
        }
    }

    pub fn draw_ik_effectors(&self, position_x: f32, position_y: f32, scale: f32) {
        for bone_id in self.ik.iter().map(|it| self.get_bone_by_name(&it.target).unwrap()) {
            let origin = self.pose_matrices[bone_id] * nalgebra::Point3::new(0.0, 0.0, 1.0);
            draw_circle(
                position_x + origin.x * scale,
                position_y + origin.y * scale,
                5.0,
                RED
            );
        }
    }

    pub fn draw_bones(&self, position_x: f32, position_y: f32, scale: f32) {
        for bone_id in 0..self.bones.len() {
            let bone = &self.bones[bone_id];
            {
                let bone_color = COLORS[bone_id % COLORS.len()];
                let origin = self.pose_matrices[bone_id] * nalgebra::Point3::new(0.0, 0.0, 1.0);
                let dir = self.pose_matrices[bone_id] * nalgebra::Point3::new(bone.length, 0.0, 1.0);
                draw_line(
                    position_x + origin.x * scale,
                    position_y + origin.y * scale,
                    position_x + dir.x * scale,
                    position_y + dir.y * scale,
                    3.0,
                    bone_color,
                );
            }
        }
    }
}

pub struct CubicBezierRegion {
    start_x: f32,
    start_y: f32,
    handle_0_x: f32,
    handle_0_y: f32,
    handle_1_x: f32,
    handle_1_y: f32,
    end_x: f32,
    end_y: f32
}
impl CubicBezierRegion {
    pub fn get_approx_bezier(buffer: &mut Vec<Self>, slice: &[f32]) -> [f32; 16] {
        Self::fill_buffer_from_slice(buffer, slice);
        let mut result = [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ];
        for i in 1..15 {
            let x = i as f32 / 15.0;
            let subregion = buffer.iter().find(|it| it.start_x <= x && x <= it.end_x).unwrap();
            result[i] = subregion.sample_at(subregion.find_t(x));
        }
        result
    }

    pub fn fill_buffer_from_slice(buffer: &mut Vec<Self>, slice: &[f32]) {
        buffer.clear();
        let (start_x, start_y, end_x, end_y) = (0.0, 0.0, 1.0, 1.0);
        let mut left_x = start_x;
        let mut left_y = start_y;
        for offset in (0..slice.len()).step_by(6) {
            let (right_x, right_y) = if slice.len() - offset == 4 {
                (end_x, end_y)
            } else {
                (slice[offset + 4], slice[offset + 5])
            };
            buffer.push(
                Self {
                    start_x: left_x,
                    start_y: left_y,
                    handle_0_x: slice[offset],
                    handle_0_y: slice[offset + 1],
                    handle_1_x: slice[offset + 2],
                    handle_1_y: slice[offset + 3],
                    end_x: right_x,
                    end_y: right_y
                }
            )
        }
    }

    pub fn find_t(&self, x: f32) -> f32 {
        const EPS: f32 = 0.01;
        const LITTLE_STEP: f32 = 0.00001;
        let (mut l, mut r) = (0.0, LITTLE_STEP);
        let (mut sample_l, mut sample_r) = (
            CubicBezierRegion::cubic_resolve(
                l,
                self.start_x,
                self.handle_0_x,
                self.handle_1_x,
                self.end_x
            ),
            CubicBezierRegion::cubic_resolve(
                r,
                self.start_x,
                self.handle_0_x,
                self.handle_1_x,
                self.end_x
            )
        );
        while !(sample_l <= x && sample_r >= x) {
            l += LITTLE_STEP;
            r += LITTLE_STEP;
            sample_l = CubicBezierRegion::cubic_resolve(
                l,
                self.start_x,
                self.handle_0_x,
                self.handle_1_x,
                self.end_x
            );
            sample_r = CubicBezierRegion::cubic_resolve(
                r,
                self.start_x,
                self.handle_0_x,
                self.handle_1_x,
                self.end_x
            );
        }
        let mut t = (l + r) / 2.0;
        let mut step = (r - l) * 0.25;
        let mut sample = CubicBezierRegion::cubic_resolve(
            t,
            self.start_x,
            self.handle_0_x,
            self.handle_1_x,
            self.end_x
        );
        while (sample - x).abs() > EPS {
            t += if sample > x { step } else { -step };
            step *= 0.5;
            sample = CubicBezierRegion::cubic_resolve(
                t,
                self.start_x,
                self.handle_0_x,
                self.handle_1_x,
                self.end_x
            );
        }
        t
    }

    pub fn sample_at(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        CubicBezierRegion::cubic_resolve(
            t,
            self.start_y,
            self.handle_0_y,
            self.handle_1_y,
            self.end_y
        )
    }

    fn cubic_resolve(t: f32, k1: f32, k2: f32, k3: f32, k4: f32) -> f32 {
        let (a, b, c) = ( k1 + (k2 - k1) * t, k2 + (k3 - k2) * t, k3 + (k4 - k3) * t);
        let (d, e) = (a + (b - a) * t, b + (c - b) * t);
        d + (e - d) * t
    }
}

#[derive(Debug)]
pub enum TweenEasing {
    None,
    Linear,
    QuadraticIn,
    QuadraticOut,
    QuadraticInOut,
    FreeCurve([f32; 16])
}
impl TweenEasing {
    pub fn parse(bezier_buffer: &mut Vec<CubicBezierRegion>, tween_easing: f32, curve_samples: &[f32]) -> Self {
        const EPS: f32 = 0.00001;
        if curve_samples.len() > 0 {
            Self::FreeCurve(CubicBezierRegion::get_approx_bezier(bezier_buffer, curve_samples))
        } else {
            if (tween_easing - 2.0).abs() <= EPS {
                Self::None
            } else if tween_easing.abs() <= EPS {
                Self::Linear
            } else if tween_easing < 0.0 {
                Self::QuadraticIn
            } else if tween_easing <= 1.0 {
                Self::QuadraticOut
            } else {
                Self::QuadraticInOut
            }
        }
    }
    pub fn interpolate(&self, a: f32, b: f32, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            TweenEasing::None => a + (b - a) * t,
            TweenEasing::Linear => a + (b - a) * t,
            TweenEasing::QuadraticIn => a + (b - a) * t * t,
            TweenEasing::QuadraticOut => b - (b - a) * (1.0 - t).powi(2),
            TweenEasing::QuadraticInOut => a + (b - a) *
                if t <= 0.5 {
                    (t * 2.0).powi(2) * 0.5
                } else {
                    -1.0 - 2.0 * t * (t - 2.0)
                },
            TweenEasing::FreeCurve(samples) => {
                let region_id = t * 15.0;
                let t = region_id.fract();
                let region_id = region_id.trunc() as usize;
                let t = if region_id >= 15 {
                    samples[15]
                } else {
                    let left = samples[region_id];
                    let right = samples[region_id + 1];
                    left + (right - left) * t
                };
                a + (b - a) * t
            }
        }
    }
}

pub trait Sample: Copy + Debug {
    fn interpolate(
        &self,
        other: Self,
        start_tick: usize,
        end_tick: usize,
        current_tick: usize,
        tween_easing: &TweenEasing
    ) -> Self;
}


#[derive(Debug)]
pub struct SamplingRegion<T: Sample> {
    pub start_sample: T,
    pub end_sample: T,
    pub start_tick: usize,
    pub end_tick: usize,
    pub tween_easing: TweenEasing
}

impl<T: Sample> SamplingRegion<T> {
    pub fn interpolate(&self, current_tick: usize) -> T {
        self.start_sample.interpolate(
            self.end_sample,
            self.start_tick,
            self.end_tick,
            current_tick,
            &self.tween_easing
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RotationSample {
    pub theta: f32
}
impl Sample for RotationSample {
    fn interpolate(
        &self,
        other: Self,
        start_tick: usize,
        end_tick: usize,
        current_tick: usize,
        tween_easing: &TweenEasing
    ) -> Self {
        debug_assert!(start_tick <= end_tick);
        let a = if end_tick == start_tick {
            1.0
        } else {
            (current_tick - start_tick) as f32 / (end_tick - start_tick) as f32
        };
        Self {
            theta: tween_easing.interpolate(self.theta, other.theta, a)
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TransitionSample {
    pub x: f32,
    pub y: f32
}
impl Sample for TransitionSample {
    fn interpolate(
        &self,
        other: Self,
        start_tick: usize,
        end_tick: usize,
        current_tick: usize,
        tween_easing: &TweenEasing
    ) -> Self {
        debug_assert!(start_tick <= end_tick);
        let a = if end_tick == start_tick {
            1.0
        } else {
            (current_tick - start_tick) as f32 / (end_tick - start_tick) as f32
        };
        Self {
            x: tween_easing.interpolate(self.x, other.x, a),
            y: tween_easing.interpolate(self.y, other.y, a)
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ScalingSample {
    pub scale_x: f32,
    pub scale_y: f32
}
impl Sample for ScalingSample {
    fn interpolate(
        &self,
        other: Self,
        start_tick: usize,
        end_tick: usize,
        current_tick: usize,
        tween_easing: &TweenEasing
    ) -> Self {
        debug_assert!(start_tick <= end_tick);
        let a = if end_tick == start_tick {
            1.0
        } else {
            (current_tick - start_tick) as f32 / (end_tick - start_tick) as f32
        };
        Self {
            scale_x: tween_easing.interpolate(self.scale_x, other.scale_x, a),
            scale_y: tween_easing.interpolate(self.scale_y, other.scale_y, a)
        }
    }
}

#[derive(Debug)]
pub struct AnimationTrack<T: Sample> {
    pub bone_id: usize,
    pub regions: Vec<SamplingRegion<T>>
}

#[derive(Debug)]
pub struct AnimationData {
    pub name: String,
    pub duration_in_ticks: usize,
    pub play_times: usize,
    pub rotation_tracks: Vec<AnimationTrack<RotationSample>>,
    pub transition_tracks: Vec<AnimationTrack<TransitionSample>>,
    pub scaling_tracks: Vec<AnimationTrack<ScalingSample>>,
}

pub struct DragonBonesData {
    armatures: HashMap<String, RuntimeArmature>,
}

impl DragonBonesData {
    pub fn load(
        skeleton_file_bytes: &[u8],
        atlas_file_bytes: &[u8],
        texture_file_bytes: &[u8],
    ) -> Self {
        let ctx = unsafe {
            let InternalGlContext { quad_context: ctx, .. } = get_internal_gl();
            ctx
        };

        let texture = load_texture(ctx, texture_file_bytes);
        let atlas = Atlas::parse(atlas_file_bytes).unwrap();
        let skeleton_data: RawSkeletonData = serde_json::from_slice(skeleton_file_bytes).unwrap();
        let mut armatures = HashMap::new();
        for armature in skeleton_data.armatures.iter() {
            let (name, runtime_armature) = RuntimeArmature::extract(armature, &atlas, texture, skeleton_data.frame_rate);
            armatures.insert(name, runtime_armature);
        }
        Self { armatures }
    }

    pub fn instantiate_armature(&self, armature_name: &str) -> Option<RuntimeArmature> {
        self.armatures.get(armature_name).map(|it| it.instantiate())
    }
}

pub struct BonesMut<'a> {
    armature: &'a mut RuntimeArmature,
}

impl<'a> core::ops::Index<usize> for BonesMut<'a> {
    type Output = NamelessBone;
    fn index(&self, index: usize) -> &Self::Output {
        &self.armature.bones[index]
    }
}

impl<'a> IndexMut<usize> for BonesMut<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let node_id = self.armature.tree_handles[index];
        for node_id in node_id.descendants(&self.armature.bone_tree) {
            self.armature.buffer_deque.push_back(node_id);
        }
        while let Some(node_id) = self.armature.buffer_deque.pop_front() {
            let mut node = self.armature.bone_tree.get_mut(node_id).unwrap().get_mut();
            node.is_dirty = true;
        }
        &mut self.armature.bones[index]
    }
}

fn load_texture(ctx: &mut Context, texture_bytes: &[u8]) -> macroquad::texture::Texture2D {
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
                filter: FilterMode::Linear,
                wrap: TextureWrap::Clamp,
            },
        )
    )
}

#[cfg(test)]
mod tests {
    use crate::runtime::TweenEasing;

    #[test]
    fn test_easings() {
        let easing = TweenEasing::Linear;
        let test_1 = easing.interpolate(0.0, 3.0, 0.0);
        let test_2 = easing.interpolate(0.0, 3.0, 1.0);
        let test_3 = easing.interpolate(0.0, 3.0, 0.5);
        assert_eq!(0.0, test_1);
        assert_eq!(3.0, test_2);
        assert_eq!(1.5, test_3);

        let easing = TweenEasing::QuadraticIn;
        let test_1 = easing.interpolate(0.0, 3.0, 0.0);
        let test_2 = easing.interpolate(0.0, 3.0, 1.0);
        let test_3 = easing.interpolate(0.0, 3.0, 0.5);
        assert_eq!(0.0, test_1);
        assert_eq!(3.0, test_2);
        assert_ne!(1.5, test_3);
        assert!(test_3 < 1.5);

        let easing = TweenEasing::QuadraticOut;
        let test_1 = easing.interpolate(0.0, 3.0, 0.0);
        let test_2 = easing.interpolate(0.0, 3.0, 1.0);
        let test_3 = easing.interpolate(0.0, 3.0, 0.5);
        assert_eq!(0.0, test_1);
        assert_eq!(3.0, test_2);
        assert_ne!(1.5, test_3);
        assert!(test_3 > 1.5);

        let easing = TweenEasing::QuadraticInOut;
        let test_1 = easing.interpolate(0.0, 3.0, 0.0);
        let test_2 = easing.interpolate(0.0, 3.0, 1.0);
        let test_3 = easing.interpolate(0.0, 3.0, 0.5);
        assert_eq!(0.0, test_1);
        assert_eq!(3.0, test_2);
        assert_eq!(1.5, test_3);
    }
}
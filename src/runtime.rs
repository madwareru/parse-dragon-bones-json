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
use std::hash::Hash;
use serde::Deserialize;
use std::cmp::Ordering;
use nalgebra::{Matrix3};

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

#[derive(Copy, Clone)]
pub enum DrawFlip {
    None,
    Flipped
}

#[derive(Copy, Clone, PartialEq)]
struct RenderQueueEntry {
    drawable_id: usize,
    tint: Color,
    implicit_draw_order: usize,
    explicit_draw_order: i32,
}
impl PartialOrd for RenderQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.explicit_draw_order != other.explicit_draw_order {
            self.explicit_draw_order.partial_cmp(&other.explicit_draw_order)
        } else {
            self.implicit_draw_order.partial_cmp(&other.implicit_draw_order)
        }
    }
}
impl Eq for RenderQueueEntry { }
impl Ord for RenderQueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.explicit_draw_order != other.explicit_draw_order {
            self.explicit_draw_order.cmp(&other.explicit_draw_order)
        } else {
            self.implicit_draw_order.cmp(&other.implicit_draw_order)
        }
    }
}

pub struct DragonBonesRuntime {
    batcher: BufferedDrawBatcher,
    render_queue: Vec<RenderQueueEntry>
}
impl DragonBonesRuntime {
    pub fn new() -> Self {
        Self {
            batcher: BufferedDrawBatcher::new(),
            render_queue: Vec::new()
        }
    }
}

pub struct BufferedDrawBatcher {
    vertex_buffer: Vec<Vertex>,
    index_buffer: Vec<u16>,
}

impl BufferedDrawBatcher {
    fn new() -> Self {
        Self {
            vertex_buffer: Vec::new(),
            index_buffer: Vec::new(),
        }
    }

    fn renderize_next_triangles(
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
    fn draw(
        &self,
        draw_batcher: &mut BufferedDrawBatcher,
        pose_matrices: &[nalgebra::Matrix3<f32>],
        diff_matrices: &[nalgebra::Matrix3<f32>],
        tint: Color,
        position_x: f32,
        position_y: f32,
        scale: f32,
        x_flipped: bool
    );
    fn instantiate(&self) -> Box<dyn Drawable>;
}

#[derive(Copy, Clone)]
pub struct BBoxDrawable { // just a stub. Maybe some day we will render it, but not now
    _nop: u8
}
impl BBoxDrawable {
    pub fn new() -> Self { Self{ _nop: 0 } }
}
impl Drawable for BBoxDrawable {
    fn draw(
        &self,
        _draw_batcher: &mut BufferedDrawBatcher,
        _pose_matrices: &[Matrix3<f32>],
        _diff_matrices: &[Matrix3<f32>],
        _tint: Color,
        _position_x: f32,
        _position_y: f32,
        _scale: f32,
        _x_flipped: bool
    ) {}
    fn instantiate(&self) -> Box<dyn Drawable> {
        Box::new(Self{ _nop: 0 })
    }
}

#[derive(Clone)]
pub struct MeshDrawable {
    mesh_data: Arc<PurifiedMeshData<u16>>,
    texture: Texture2D,
    atlas_size: [f32; 2],
    atlas_sub_texture: SubTexture
}

impl MeshDrawable {
    pub fn new(
        mesh_data: &PurifiedMeshData<u16>,
        texture: Texture2D,
        atlas_size: [f32; 2],
        atlas_sub_texture: SubTexture
    ) -> Self {
        Self {
            mesh_data: Arc::new(mesh_data.clone()),
            texture,
            atlas_size,
            atlas_sub_texture
        }
    }
}

impl Drawable for MeshDrawable {
    fn draw(
        &self,
        draw_batcher: &mut BufferedDrawBatcher,
        _pose_matrices: &[nalgebra::Matrix3<f32>],
        diff_matrices: &[nalgebra::Matrix3<f32>],
        tint: Color,
        position_x: f32,
        position_y: f32,
        scale: f32,
        flipped: bool
    ) {
        let x_scale = if flipped { -scale} else { scale };
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
                        position_x + vert_new.x * x_scale,
                        position_y + vert_new.y * scale,
                        0.0,
                        self.atlas_sub_texture.rect.x / self.atlas_size[0]
                            + uv[0] * self.atlas_sub_texture.rect.width / self.atlas_size[0],
                        self.atlas_sub_texture.rect.y / self.atlas_size[1]
                            + uv[1] * self.atlas_sub_texture.rect.height / self.atlas_size[1],
                        tint
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
        Box::new(Self {
            mesh_data,
            texture,
            atlas_size,
            atlas_sub_texture
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
}

impl ImageDrawable {
    pub fn new(
        texture: Texture2D,
        atlas_size: [f32; 2],
        atlas_sub_texture: SubTexture,
        parent_bone_id: usize,
        transform: RawTransform,
        pivot: crate::shared_types::Point
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
            uvs
        }
    }
}

impl Drawable for ImageDrawable {
    fn draw(
        &self,
        draw_batcher: &mut BufferedDrawBatcher,
        pose_matrices: &[nalgebra::Matrix3<f32>],
        _diff_matrices: &[nalgebra::Matrix3<f32>],
        tint: Color,
        position_x: f32,
        position_y: f32,
        scale: f32,
        flipped: bool
    ) {
        let x_scale = if flipped { -scale} else { scale };
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
                        position_x + vert_new.x * x_scale,
                        position_y + vert_new.y * scale,
                        0.0,
                        uv.0,
                        uv.1,
                        tint
                    )
                });
        draw_batcher.renderize_next_triangles(verts, indices, Some(self.texture));
    }

    fn instantiate(&self) -> Box<dyn Drawable> {
        let texture = self.texture;
        let atlas_size = self.atlas_size;
        let atlas_sub_texture = self.atlas_sub_texture.clone();
        Box::new(Self {
            texture,
            atlas_size,
            atlas_sub_texture,
            parent_bone_id: self.parent_bone_id,
            transform: self.transform,
            vertices: self.vertices.clone(),
            uvs: self.uvs
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

#[derive(Copy, Clone, Debug, PartialEq)]
enum Tick {
    Current,
    FadeOut
}

#[derive(Clone)]
struct SharedArmatureInfo {
    ms_per_tick: f32,
    ik: Arc<Vec<IkInfo>>,
    bone_lookup: Arc<HashMap<String, usize>>,
    slot_lookup: Arc<HashMap<String, usize>>,
    rest_pose_bones: Arc<Vec<NamelessBone>>,
    animations: Arc<Vec<AnimationData>>,
    initial_matrices: Arc<Vec<nalgebra::Matrix3<f32>>>,
    start_animation_id: usize,
    initial_slot_info: Arc<Vec<SlotInfo>>,
}

#[derive(Clone)]
struct AnimationInfo {
    ticked_time: f32,
    bones: Vec<NamelessBone>,
    current_animation_id: usize,
    current_animation_ticks: usize
}

#[derive(Copy, Clone)]
struct FadeOutInfo {
    duration_in_frames: usize,
    current_frame: usize
}

#[derive(Copy, Clone)]
pub struct AdditiveAnimationInfo {
    ticked_time: f32,
    animation_id: usize,
    animation_ticks: usize
}

#[derive(Clone)]
struct SlotInfo {
    tint_a: i32,
    tint_r: i32,
    tint_g: i32,
    tint_b: i32,
    draw_order: i32,
    display_id: Option<usize>,
    drawable_indices: std::ops::Range<usize>,
}

pub struct RuntimeArmature {
    shared_info: SharedArmatureInfo,
    fade_out_animation_info: AnimationInfo,
    current_animation_info: AnimationInfo,
    fade_out: Option<FadeOutInfo>,
    additive_animations: Vec<AdditiveAnimationInfo>,

    bone_tree: Arena<BoneInfo>,
    tree_handles: Vec<indextree::NodeId>,

    bones: Vec<NamelessBone>,
    pose_matrices: Vec<nalgebra::Matrix3<f32>>,
    diff_matrices: Vec<nalgebra::Matrix3<f32>>,

    slots: Vec<SlotInfo>,
    drawables: Vec<Box<dyn Drawable>>,
    buffer_deque: VecDeque<indextree::NodeId>
}

impl RuntimeArmature {
    pub fn get_bone_by_name(&self, bone_name: &str) -> Option<usize> {
        self.shared_info.bone_lookup.get(bone_name).map(|&it| it)
    }
}

impl RuntimeArmature {
    pub fn get_bone_world_orientation(
        &self,
        bone_id: usize,
        x_flip: DrawFlip
    ) -> (f32, f32) {
        let x_scale = match x_flip {
            DrawFlip::None => 1.0,
            DrawFlip::Flipped => -1.0
        };
        let dir = self.pose_matrices[bone_id] * nalgebra::Point3::new(self.bones[bone_id].length, 0.0, 1.0);
        let dir = (
            dir.x * x_scale,
            dir.y
        );
        let len = (dir.0 * dir.0 + dir.1 * dir.1).sqrt();
        (
            dir.0 / len,
            dir.1 / len
        )
    }

    pub fn get_bone_world_position(
        &self,
        bone_id: usize,
        position_x: f32,
        position_y: f32,
        scale: f32,
        x_flip: DrawFlip
    ) -> (f32, f32) {
        let x_scale = match x_flip {
            DrawFlip::None => scale,
            DrawFlip::Flipped => -scale
        };
        let origin = self.pose_matrices[bone_id] * nalgebra::Point3::new(0.0, 0.0, 1.0);
        (
            position_x + origin.x * x_scale,
            position_y + origin.y * scale
        )
    }

    pub fn instantiate(&self) -> Self {
        let bones = self.bones.clone();
        let pose_matrices = self.pose_matrices.clone();
        let diff_matrices = self.diff_matrices.clone();
        let drawables = self.drawables.iter().map(|it| it.instantiate()).collect();
        Self {
            shared_info: self.shared_info.clone(),
            fade_out_animation_info: self.fade_out_animation_info.clone(),
            current_animation_info: self.current_animation_info.clone(),
            additive_animations: self.additive_animations.clone(),
            fade_out: self.fade_out,
            bones,
            slots: self.slots.clone(),
            drawables,
            pose_matrices,
            diff_matrices,
            bone_tree: self.bone_tree.clone(),
            tree_handles: self.tree_handles.clone(),
            buffer_deque: VecDeque::new(),
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

                let mut slot_vec = Vec::with_capacity(slots.len());
                let mut slot_lookup = HashMap::new();
                for bone in bones.iter() {
                    bone_lookup.insert(bone.name.clone(), bone_vec.len());
                    bone_vec.push(NamelessBone::from((bone, &bones[..])));
                }

                let mut initial_tints = Vec::new();
                let mut initial_display_ids = Vec::new();
                for i in 0..slots.len() {
                    slot_lookup.insert(slots[i].name.clone(), slot_vec.len());
                    initial_tints.push(
                        (
                            slots[i].color_transform.alpha_multiplier,
                            slots[i].color_transform.red_multiplier,
                            slots[i].color_transform.green_multiplier,
                            slots[i].color_transform.blue_multiplier
                        )
                    );
                    let display_id = if slots[i].display_id >= 0 {
                        Some(slots[i].display_id as usize)
                    } else {
                        None
                    };
                    initial_display_ids.push(display_id);
                    slot_vec.push(SlotInfo {
                        tint_a: slots[i].color_transform.alpha_multiplier,
                        tint_r: slots[i].color_transform.red_multiplier,
                        tint_g: slots[i].color_transform.green_multiplier,
                        tint_b: slots[i].color_transform.blue_multiplier,
                        draw_order: 0,
                        display_id,
                        drawable_indices: std::ops::Range::default()
                    });
                }
                for skin in skins.iter() {
                    for slot in skin.slots.iter() {
                        let (parent_bone_name, slot_id) =
                            slots.iter().enumerate().find_map(|(id, it)| {
                                if it.name.eq(&slot.name) { Some((&it.parent as &str, id as i32)) } else { None }
                            }).unwrap_or(("", i32::MIN));
                        let parent_bone_id = (0..bones.len()).find(|&id| bones[id].name.eq(parent_bone_name));
                        let mut first_drawable = true;
                        for display in slot.displays.iter() {
                            let sub_texture = display.get_rect(atlas);
                            if slot_id >= 0 {
                                if first_drawable {
                                    first_drawable = false;
                                    slot_vec[slot_id as usize].drawable_indices.start = drawables.len();
                                    slot_vec[slot_id as usize].drawable_indices.end = drawables.len() + 1;
                                } else {
                                    slot_vec[slot_id as usize].drawable_indices.end = drawables.len() + 1;
                                }
                            }
                            if let Some(mesh) = PurifiedMeshData::<u16>::try_from(display, bones.len()) {
                                drawables.push(Box::new(
                                    MeshDrawable::new(
                                        &mesh,
                                        texture,
                                        [atlas.width as f32, atlas.height as f32],
                                        sub_texture.unwrap()
                                    )
                                ));
                                continue;
                            }
                            if let &RawDisplay::Image { pivot, transform, .. } = display {
                                drawables.push(Box::new(
                                    ImageDrawable::new(
                                        texture,
                                        [atlas.width as f32, atlas.height as f32],
                                        sub_texture.unwrap(),
                                        parent_bone_id.unwrap(),
                                        transform,
                                        pivot
                                    )
                                ));
                                continue;
                            }
                            drawables.push(Box::new(BBoxDrawable::new()));
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
                        scaling_tracks: Vec::new(),
                        display_id_animation_tracks: Vec::new(),
                        color_tint_animation_tracks: Vec::new(),
                        draw_order_animation_track: DrawOrderAnimationTrack { regions: Vec::new() }
                    };
                    for slot_timeline in anim.slot_timelines.iter() {
                        let slot_id = *slot_lookup.get(&slot_timeline.name).unwrap();
                        if slot_timeline.display_frames.len() > 0 {
                            let mut animation_track = DisplayIdAnimationTrack {
                                slot_id,
                                regions: Vec::new()
                            };
                            let mut tick_now: usize = 0;
                            for frame_timeline in slot_timeline.display_frames.iter() {
                                animation_track.regions.push(SlotDisplayIdRegion {
                                    display_id: if frame_timeline.value >= 0 {
                                        Some(frame_timeline.value as usize)
                                    } else {
                                        None
                                    },
                                    start_tick: tick_now,
                                    end_tick: tick_now + frame_timeline.duration as usize
                                });
                                tick_now += frame_timeline.duration as usize;
                            }
                            animation_data.display_id_animation_tracks.push(animation_track);
                        }
                        if slot_timeline.colors.len() > 0 {
                            let mut animation_track = ColorTintAnimationTrack {
                                slot_id,
                                regions: Vec::new()
                            };
                            let mut tick_now: usize = 0;
                            if slot_timeline.colors.len() == 1 {
                                let frame = &slot_timeline.colors[0];
                                animation_track.regions.push(
                                    SamplingRegion {
                                        start_sample: TintSample{
                                            a: frame.color.alpha_multiplier,
                                            r: frame.color.red_multiplier,
                                            g: frame.color.green_multiplier,
                                            b: frame.color.blue_multiplier,
                                        },
                                        end_sample: TintSample{
                                            a: frame.color.alpha_multiplier,
                                            r: frame.color.red_multiplier,
                                            g: frame.color.green_multiplier,
                                            b: frame.color.blue_multiplier,
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
                            } else {
                                for i in 0..slot_timeline.colors.len()-1 {
                                    let frame = &slot_timeline.colors[i];
                                    let next_frame = &slot_timeline.colors[i+1];
                                    animation_track.regions.push(
                                        SamplingRegion {
                                            start_sample: TintSample{
                                                a: frame.color.alpha_multiplier,
                                                r: frame.color.red_multiplier,
                                                g: frame.color.green_multiplier,
                                                b: frame.color.blue_multiplier,
                                            },
                                            end_sample: TintSample{
                                                a: next_frame.color.alpha_multiplier,
                                                r: next_frame.color.red_multiplier,
                                                g: next_frame.color.green_multiplier,
                                                b: next_frame.color.blue_multiplier,
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
                                    tick_now += slot_timeline.colors[i].duration as usize;
                                }
                            }
                            animation_data.color_tint_animation_tracks.push(animation_track);
                        }
                    }
                    {
                        let mut tick_now: usize = 0;
                        for draw_order_frame in anim.z_order_timeline.frames.iter() {
                            let mut entries: Vec<DrawOrderEntry> = Vec::new();
                            for offset in (0..draw_order_frame.z_order.len()).step_by(2) {
                                entries.push(
                                    DrawOrderEntry {
                                        slot_id: draw_order_frame.z_order[offset] as usize,
                                        draw_order: draw_order_frame.z_order[offset + 1]
                                    }
                                )
                            }
                            animation_data.draw_order_animation_track.regions.push(
                                DrawOrderRegion {
                                    entries,
                                    start_tick: tick_now,
                                    end_tick: tick_now + draw_order_frame.duration as usize
                                }
                            );
                            tick_now += draw_order_frame.duration as usize;
                        }
                    }
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
                                if bone_timeline.rotation_frames.len() == 1 {
                                    let frame = &bone_timeline.rotation_frames[0];
                                    let rotation = bone_vec[bone_id].transform.rotation
                                        + frame.rotate.to_radians()
                                        * if frame.clockwise == 1 { -1.0 } else { 1.0 };
                                    animation_track.regions.push(
                                        SamplingRegion {
                                            start_sample: RotationSample{ theta: rotation },
                                            end_sample: RotationSample{ theta: rotation },
                                            start_tick: tick_now,
                                            end_tick: tick_now + frame.duration as usize,
                                            tween_easing: TweenEasing::parse(
                                                &mut bezier_buffer,
                                                frame.tween_easing,
                                                &frame.curve
                                            )
                                        }
                                    );
                                } else {
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
                                }
                                animation_data.rotation_tracks.push(animation_track);
                            }
                            if bone_timeline.translate_frames.len() > 0 {
                                let mut tick_now: usize = 0;
                                let mut animation_track = AnimationTrack {
                                    bone_id,
                                    regions: Vec::new()
                                };
                                if bone_timeline.translate_frames.len() == 1 {
                                    let frame = &bone_timeline.translate_frames[0];
                                    let translation_x = bone_vec[bone_id].transform.x + frame.x;
                                    let translation_y = bone_vec[bone_id].transform.y + frame.y;
                                    animation_track.regions.push(
                                        SamplingRegion {
                                            start_sample: TransitionSample{
                                                x: translation_x,
                                                y: translation_y
                                            },
                                            end_sample: TransitionSample{
                                                x: translation_x,
                                                y: translation_y
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
                                } else {
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
                                }
                                animation_data.transition_tracks.push(animation_track);
                            }
                            if bone_timeline.scale_frames.len() > 0 {
                                let mut tick_now: usize = 0;
                                let mut animation_track = AnimationTrack {
                                    bone_id,
                                    regions: Vec::new()
                                };
                                if bone_timeline.scale_frames.len() == 1 {
                                    let frame = &bone_timeline.scale_frames[0];
                                    let scale_x = frame.x;
                                    let scale_y = frame.y;
                                    animation_track.regions.push(
                                        SamplingRegion {
                                            start_sample: ScalingSample{
                                                scale_x,
                                                scale_y
                                            },
                                            end_sample: ScalingSample{
                                                scale_x,
                                                scale_y
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
                                } else {
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

                let animation_info = AnimationInfo {
                    ticked_time: 0.0,
                    bones: bone_vec.clone(),
                    current_animation_id: start_animation_id,
                    current_animation_ticks: 0
                };

                (
                    name.clone(),
                    Self {
                        shared_info: SharedArmatureInfo {
                            ms_per_tick: 1.0 / frame_rate as f32,
                            rest_pose_bones: Arc::new(bone_vec.clone()),
                            initial_slot_info: Arc::new(slot_vec.clone()),
                            initial_matrices: Arc::new(initial_matrices),
                            bone_lookup: Arc::new(bone_lookup),
                            slot_lookup: Arc::new(slot_lookup),
                            animations: Arc::new(animations_vec),
                            start_animation_id,
                            ik
                        },
                        fade_out_animation_info: animation_info.clone(),
                        current_animation_info: animation_info,
                        additive_animations: Vec::new(),
                        fade_out: None,
                        bones: bone_vec,
                        slots: slot_vec,
                        drawables,
                        pose_matrices,
                        diff_matrices,
                        bone_tree,
                        tree_handles,
                        buffer_deque: VecDeque::new()
                    }
                )
            }
        }
    }

    pub fn stack_additive_animation(&mut self, animation_name: &str) {
        let animation_id = (0..self.shared_info.animations.len()).find(|&id| {
            self.shared_info.animations[id].name.eq(animation_name)
        });
        if let Some(animation_id) = animation_id {
            self.additive_animations.push(
                AdditiveAnimationInfo {
                    ticked_time: 0.0,
                    animation_id,
                    animation_ticks: 0
                }
            )
        }
    }

    pub fn goto_and_play(&mut self, animation_name: &str, fade_out: Option<usize>) {
        let animation_id = (0..self.shared_info.animations.len()).find(|&id| {
            self.shared_info.animations[id].name.eq(animation_name)
        });
        if let Some(animation_id) = animation_id {
            std::mem::swap(&mut self.current_animation_info, &mut self.fade_out_animation_info);
            self.fade_out = fade_out.map(|it| FadeOutInfo { duration_in_frames: it, current_frame: 0 });

            self.current_animation_info.ticked_time = self.shared_info.ms_per_tick;
            self.current_animation_info.current_animation_id = animation_id;
            self.current_animation_info.current_animation_ticks = 0;
            for i in 0..self.bones.len() {
                self.current_animation_info.bones[i].transform = self.shared_info.rest_pose_bones[i].transform;
            }
            for i in 0..self.slots.len() {
                self.slots[i] = self.shared_info.initial_slot_info[i].clone();
            }
        }
    }

    fn shared_animation_update(&mut self, dt: f32) {
        self.current_animation_info.ticked_time += dt;
        while self.current_animation_info.ticked_time >= self.shared_info.ms_per_tick {
            self.current_animation_info.ticked_time -= self.shared_info.ms_per_tick;
            self.current_animation_info.current_animation_ticks += 1;
        }
        self.tick_animation(Tick::Current);
        self.fade_out = match self.fade_out {
            None => {
                {
                    let bones_amount = self.current_animation_info.bones.len();
                    for i in 0..bones_amount {
                        let bone = self.current_animation_info.bones[i];
                        let mut bones_mut = BonesMut { armature: self };
                        bones_mut[i] = bone;
                    }
                }
                None
            },
            Some(fade_out) => {
                if fade_out.current_frame >= fade_out.duration_in_frames {
                    {
                        let bones_amount = self.current_animation_info.bones.len();
                        for i in 0..bones_amount {
                            let bone = self.current_animation_info.bones[i];
                            let mut bones_mut = BonesMut { armature: self };
                            bones_mut[i] = bone;
                        }
                    }
                    None
                } else {
                    let current_delta = fade_out.current_frame as f32 / fade_out.duration_in_frames as f32;
                    self.fade_out_animation_info.ticked_time += dt;
                    while self.fade_out_animation_info.ticked_time >= self.shared_info.ms_per_tick {
                        self.fade_out_animation_info.ticked_time -= self.shared_info.ms_per_tick;
                        self.fade_out_animation_info.current_animation_ticks += 1;
                        self.tick_animation(Tick::FadeOut);
                    }
                    {
                        let bones_amount = self.current_animation_info.bones.len();
                        for i in 0..bones_amount {
                            let bone = self.current_animation_info.bones[i];
                            let bone_fade = self.fade_out_animation_info.bones[i];
                            let mut bones_mut = BonesMut { armature: self };
                            bones_mut[i].transform.rotation = TweenEasing::Linear.interpolate(
                                bone_fade.transform.rotation,
                                bone.transform.rotation,
                                current_delta
                            );
                            bones_mut[i].transform.x = TweenEasing::Linear.interpolate(
                                bone_fade.transform.x,
                                bone.transform.x,
                                current_delta
                            );
                            bones_mut[i].transform.y = TweenEasing::Linear.interpolate(
                                bone_fade.transform.y,
                                bone.transform.y,
                                current_delta
                            );
                            bones_mut[i].transform.scale_x = TweenEasing::Linear.interpolate(
                                bone_fade.transform.scale_x,
                                bone.transform.scale_x,
                                current_delta
                            );
                            bones_mut[i].transform.scale_y = TweenEasing::Linear.interpolate(
                                bone_fade.transform.scale_y,
                                bone.transform.scale_y,
                                current_delta
                            );
                        }
                    }
                    Some( FadeOutInfo {current_frame: fade_out.current_frame + 1, ..fade_out} )
                }
            }
        };
        for idx in (0..self.additive_animations.len()).rev() {
            let anim_id = self.additive_animations[idx].animation_id;
            self.additive_animations[idx].ticked_time += dt;
            while self.additive_animations[idx].ticked_time >= self.shared_info.ms_per_tick {
                self.additive_animations[idx].ticked_time -= self.shared_info.ms_per_tick;
                self.additive_animations[idx].animation_ticks += 1;
            }
            if self.additive_animations[idx].animation_ticks > self.shared_info.animations[anim_id].duration_in_ticks {
                self.additive_animations.remove(idx);
                continue;
            }
            let num_rotation_tracks = self.shared_info.animations[anim_id].rotation_tracks.len();
            let num_transition_tracks = self.shared_info.animations[anim_id].transition_tracks.len();
            let num_scaling_tracks = self.shared_info.animations[anim_id].scaling_tracks.len();
            for i in 0..num_rotation_tracks {
                let (bone_id, current_frame) = {
                    let track = &self.shared_info.animations[anim_id].rotation_tracks[i];
                    (
                        track.bone_id,
                        track.regions.iter().find(|&it| {
                            it.start_tick <= self.additive_animations[idx].animation_ticks &&
                                it.end_tick >= self.additive_animations[idx].animation_ticks
                        })
                    )
                };
                if let Some(current_frame) = current_frame {
                    let interpolated_rotation = current_frame.interpolate(self.additive_animations[idx].animation_ticks);
                    let mut bones_mut = BonesMut { armature: self };
                    bones_mut[bone_id].transform.rotation = interpolated_rotation.theta;
                }
            }
            for i in 0..num_transition_tracks {
                let (bone_id, current_frame) = {
                    let track = &self.shared_info.animations[anim_id].transition_tracks[i];
                    (
                        track.bone_id,
                        track.regions.iter().find(|&it| {
                            it.start_tick <= self.additive_animations[idx].animation_ticks &&
                                it.end_tick >= self.additive_animations[idx].animation_ticks
                        })
                    )
                };
                if let Some(current_frame) = current_frame {
                    let interpolated_translation = current_frame.interpolate(self.additive_animations[idx].animation_ticks);
                    let mut bones_mut = BonesMut { armature: self };
                    bones_mut[bone_id].transform.x = interpolated_translation.x;
                    bones_mut[bone_id].transform.y = interpolated_translation.y;
                }
            }
            for i in 0..num_scaling_tracks {
                let (bone_id, current_frame) = {
                    let track = &self.shared_info.animations[anim_id].scaling_tracks[i];
                    (
                        track.bone_id,
                        track.regions.iter().find(|&it| {
                            it.start_tick <= self.additive_animations[idx].animation_ticks &&
                                it.end_tick >= self.additive_animations[idx].animation_ticks
                        })
                    )
                };
                if let Some(current_frame) = current_frame {
                    let interpolated_scale = current_frame.interpolate(self.additive_animations[idx].animation_ticks);
                    let mut bones_mut = BonesMut { armature: self };
                    bones_mut[bone_id].transform.scale_x = interpolated_scale.scale_x;
                    bones_mut[bone_id].transform.scale_y = interpolated_scale.scale_y;
                }
            }
        }
        self.update_matrices();
    }

    pub fn update_animation(&mut self, dt: f32) {
        self.shared_animation_update(dt);
        self.update_ik();
    }

    pub fn update_animation_ex(
        &mut self,
        dt: f32,
        post_process_animation: impl FnOnce(&mut BonesMut) -> (),
    ) {
        self.shared_animation_update(dt);
        {
            let mut bones_mut = BonesMut { armature: self };
            post_process_animation(&mut bones_mut);
        }
        self.update_matrices();
        self.update_ik();
    }

    fn tick_animation(&mut self, tick_kind: Tick) {
        let current_animation_info = match tick_kind {
            Tick::Current => &mut self.current_animation_info,
            Tick::FadeOut => &mut self.fade_out_animation_info,
        };
        let play_times = self.shared_info.animations[current_animation_info.current_animation_id].play_times;
        let duration_in_ticks = self.shared_info.animations[current_animation_info.current_animation_id].duration_in_ticks;
        if duration_in_ticks == 0 {
            return;
        }
        if play_times != 0 && duration_in_ticks * play_times <= current_animation_info.current_animation_ticks {
            return;
        }
        if play_times == 0 && duration_in_ticks <= current_animation_info.current_animation_ticks {
            current_animation_info.current_animation_ticks = current_animation_info.current_animation_ticks % duration_in_ticks;
        }

        if tick_kind == Tick::Current {
            // We don't need to animate slots in fade out, so this works only for current animation info
            // and does a direct modification for slots in armature
            let num_display_id_tracks = self.shared_info.animations[current_animation_info.current_animation_id].display_id_animation_tracks.len();
            let num_color_tint_tracks = self.shared_info.animations[current_animation_info.current_animation_id].color_tint_animation_tracks.len();
            let num_draw_order_regions = self.shared_info.animations[current_animation_info.current_animation_id].draw_order_animation_track.regions.len();

            for i in 0..num_display_id_tracks {
                let (slot_id, current_frame) = {
                    let track = &self.shared_info.animations[current_animation_info.current_animation_id].display_id_animation_tracks[i];
                    (
                        track.slot_id,
                        track.regions.iter().find(|&it| {
                            it.start_tick <= current_animation_info.current_animation_ticks &&
                                it.end_tick >= current_animation_info.current_animation_ticks
                        })
                    )
                };
                if let Some(current_frame) = current_frame {
                    match current_frame.display_id {
                        None => self.slots[slot_id].display_id = None,
                        Some(id) => {
                            if id + self.slots[slot_id].drawable_indices.start < self.slots[slot_id].drawable_indices.end {
                                self.slots[slot_id].display_id = Some(id);
                            }
                        }
                    }
                }
            }
            for i in 0..num_color_tint_tracks {
                let (slot_id, current_frame) = {
                    let track = &self.shared_info.animations[current_animation_info.current_animation_id].color_tint_animation_tracks[i];
                    (
                        track.slot_id,
                        track.regions.iter().find(|&it| {
                            it.start_tick <= current_animation_info.current_animation_ticks &&
                                it.end_tick >= current_animation_info.current_animation_ticks
                        })
                    )
                };
                if let Some(current_frame) = current_frame {
                    let interpolated_sample = current_frame.interpolate(current_animation_info.current_animation_ticks);
                    self.slots[slot_id].tint_a = interpolated_sample.a;
                    self.slots[slot_id].tint_r = interpolated_sample.r;
                    self.slots[slot_id].tint_g = interpolated_sample.g;
                    self.slots[slot_id].tint_b = interpolated_sample.b;
                }
            }

            if num_draw_order_regions > 0 {
                let current_frame = {
                    let track = &self.shared_info.animations[current_animation_info.current_animation_id].draw_order_animation_track;
                    track.regions.iter().find(|&it| {
                        it.start_tick <= current_animation_info.current_animation_ticks &&
                            it.end_tick >= current_animation_info.current_animation_ticks
                    })
                };
                if let Some(current_frame) = current_frame {
                    for i in 0..self.slots.len() {
                        self.slots[i].draw_order = 0;
                    }
                    for entry in current_frame.entries.iter() {
                        self.slots[entry.slot_id].draw_order = entry.draw_order;
                    }
                }
            }

        }

        let num_rotation_tracks = self.shared_info.animations[current_animation_info.current_animation_id].rotation_tracks.len();
        let num_transition_tracks = self.shared_info.animations[current_animation_info.current_animation_id].transition_tracks.len();
        let num_scaling_tracks = self.shared_info.animations[current_animation_info.current_animation_id].scaling_tracks.len();
        for i in 0..num_rotation_tracks {
            let (bone_id, current_frame) = {
                let track = &self.shared_info.animations[current_animation_info.current_animation_id].rotation_tracks[i];
                (
                    track.bone_id,
                    track.regions.iter().find(|&it| {
                        it.start_tick <= current_animation_info.current_animation_ticks &&
                        it.end_tick >= current_animation_info.current_animation_ticks
                    })
                )
            };
            if let Some(current_frame) = current_frame {
                let interpolated_rotation = current_frame.interpolate(current_animation_info.current_animation_ticks);
                current_animation_info.bones[bone_id].transform.rotation = interpolated_rotation.theta;
            }
        }
        for i in 0..num_transition_tracks {
            let (bone_id, current_frame) = {
                let track = &self.shared_info.animations[current_animation_info.current_animation_id].transition_tracks[i];
                (
                    track.bone_id,
                    track.regions.iter().find(|&it| {
                        it.start_tick <= current_animation_info.current_animation_ticks &&
                            it.end_tick >= current_animation_info.current_animation_ticks
                    })
                )
            };
            if let Some(current_frame) = current_frame {
                let interpolated_translation = current_frame.interpolate(current_animation_info.current_animation_ticks);
                current_animation_info.bones[bone_id].transform.x = interpolated_translation.x;
                current_animation_info.bones[bone_id].transform.y = interpolated_translation.y;
            }
        }
        for i in 0..num_scaling_tracks {
            let (bone_id, current_frame) = {
                let track = &self.shared_info.animations[current_animation_info.current_animation_id].scaling_tracks[i];
                (
                    track.bone_id,
                    track.regions.iter().find(|&it| {
                        it.start_tick <= current_animation_info.current_animation_ticks &&
                            it.end_tick >= current_animation_info.current_animation_ticks
                    })
                )
            };
            if let Some(current_frame) = current_frame {
                let interpolated_scale = current_frame.interpolate(current_animation_info.current_animation_ticks);
                current_animation_info.bones[bone_id].transform.scale_x = interpolated_scale.scale_x;
                current_animation_info.bones[bone_id].transform.scale_y = interpolated_scale.scale_y;
            }
        }
    }

    fn update_ik(&mut self) {
        for ik_info_id in 0..self.shared_info.ik.len() {
            let (bone_id, effector_bone_id, chain_length, bend_positive) = {
                let bone_id = self.get_bone_by_name(&self.shared_info.ik[ik_info_id].bone).unwrap();
                let effector_bone_id = self.get_bone_by_name(&self.shared_info.ik[ik_info_id].target).unwrap();
                (bone_id, effector_bone_id, self.shared_info.ik[ik_info_id].chain_length, self.shared_info.ik[ik_info_id].bend_positive)
            };

            let effector_position: nalgebra::Point3<f32> =
                self.pose_matrices[effector_bone_id] *
                    nalgebra::Point3::new(0.0, 0.0, 1.0);

            match chain_length {
                0 => {
                    let origin: nalgebra::Point3<f32> = self.pose_matrices[bone_id] * nalgebra::Point3::new(0.0, 0.0, 1.0);
                    let delta = effector_position - origin;
                    let rotation = delta.y.atan2(delta.x);
                    let mut bones_mut = BonesMut { armature: self };
                    bones_mut.set_bone_world_rotation(bone_id, rotation);
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
                        let k1 = delta.magnitude();
                        let k2: f32 = l1 * l1 - l2 * l2;

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
        self.update_matrices();
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
                self.diff_matrices[bone_id] = self.pose_matrices[bone_id] * self.shared_info.initial_matrices[bone_id];
            }
        }
    }

    pub fn set_slot_display_id(&mut self, slot_name: &str, id: Option<usize>) {
        let idx = self.shared_info.slot_lookup.get(slot_name).map(|it| *it);
        if let Some(idx) = idx {
            match id {
                None => self.slots[idx].display_id = None,
                Some(id) => {
                    if id < self.slots[idx].drawable_indices.end - self.slots[idx].drawable_indices.start {
                        self.slots[idx].display_id = Some(id);
                    }
                }
            }
        }
    }

    pub fn draw(
        &self,
        runtime: &mut DragonBonesRuntime,
        position_x: f32,
        position_y: f32,
        scale: f32,
        flip_x: DrawFlip
    ) {
        let x_flipped = match flip_x {
            DrawFlip::None => false,
            DrawFlip::Flipped => true
        };
        runtime.render_queue.clear();
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.drawable_indices.is_empty() { continue; }
            if let Some(display_id) = slot.display_id {
                let id = slot.drawable_indices.start + display_id;
                if id >= slot.drawable_indices.end {
                    continue;
                }
                let color = Color::new(
                    slot.tint_r as f32 / 100.0,
                    slot.tint_g as f32 / 100.0,
                    slot.tint_b as f32 / 100.0,
                    slot.tint_a as f32 / 100.0
                );
                runtime.render_queue.push(RenderQueueEntry {
                    tint: color,
                    drawable_id: id,
                    implicit_draw_order: i,
                    explicit_draw_order: slot.draw_order
                });
            }
        }
        runtime.render_queue.sort();
        for &entry in runtime.render_queue.iter() {
            self.drawables[entry.drawable_id].draw(
                &mut runtime.batcher,
                &self.pose_matrices,
                &self.diff_matrices,
                entry.tint,
                position_x,
                position_y,
                scale,
                x_flipped
            );
        }
    }

    pub fn draw_ik_effectors(&self, position_x: f32, position_y: f32, scale: f32) {
        for bone_id in self.shared_info.ik.iter().map(|it| self.get_bone_by_name(&it.target).unwrap()) {
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
    pub fn get_approx_bezier(buffer: &mut Vec<Self>, slice: &[f32]) -> [u8; 16] {
        Self::fill_buffer_from_slice(buffer, slice);
        let mut result = [
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 255
        ];
        for i in 1..15 {
            let x = i as f32 / 15.0;
            let subregion = buffer.iter().find(|it| it.start_x <= x && x <= it.end_x).unwrap();
            result[i] = (subregion.sample_at(subregion.find_t(x)) * 255.0).clamp(0.0, 255.0) as u8;
        }
        result
    }

    pub fn fill_buffer_from_slice(buffer: &mut Vec<Self>, slice: &[f32]) {
        buffer.clear();
        let (start_x, start_y, end_x, end_y) = (0.0, 0.0, 1.0, 1.0);
        let left_x = start_x;
        let left_y = start_y;
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
    FreeCurve([u8; 16])
}
impl TweenEasing {
    pub fn parse(bezier_buffer: &mut Vec<CubicBezierRegion>, tween_easing: f32, curve_samples: &[f32]) -> Self {
        const EPS: f32 = 0.00001;
        if curve_samples.len() >= 4 {
            Self::FreeCurve(CubicBezierRegion::get_approx_bezier(bezier_buffer, curve_samples))
        } else {
            if (tween_easing - 2.0).abs() <= EPS {
                Self::None
            } else {
                Self::Linear
            }
        }
    }
    pub fn interpolate(&self, a: f32, b: f32, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        let t = match self {
            TweenEasing::None => 0.0,
            TweenEasing::Linear => t,
            TweenEasing::FreeCurve(samples) => {
                let region_id = t * 15.0;
                let t = region_id.fract();
                let region_id = region_id.trunc() as usize;
                if region_id >= 15 {
                    1.0
                } else {
                    let left = samples[region_id] as f32 / 255.0;
                    let right = samples[region_id + 1] as f32 / 255.0;
                    left + (right - left) * t
                }
            }
        };
        a + (b - a) * t
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
pub struct SlotDisplayIdRegion {
    pub display_id: Option<usize>,
    pub start_tick: usize,
    pub end_tick: usize
}

#[derive(Debug)]
pub struct DisplayIdAnimationTrack {
    pub slot_id: usize,
    pub regions: Vec<SlotDisplayIdRegion>
}

#[derive(Debug)]
pub struct DrawOrderEntry {
    pub slot_id: usize,
    pub draw_order: i32
}

#[derive(Debug)]
pub struct DrawOrderRegion {
    pub entries: Vec<DrawOrderEntry>,
    pub start_tick: usize,
    pub end_tick: usize
}

#[derive(Debug)]
pub struct DrawOrderAnimationTrack {
    pub regions: Vec<DrawOrderRegion>
}

#[derive(Copy, Clone, Debug)]
pub struct TintSample {
    pub a: i32,
    pub r: i32,
    pub g: i32,
    pub b: i32
}
impl Sample for TintSample {
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
            a: tween_easing
                .interpolate(self.a as f32, other.a as f32, a)
                .clamp(0.0, 100.0) as i32,
            r: tween_easing
                .interpolate(self.r as f32, other.r as f32, a)
                .clamp(0.0, 100.0) as i32,
            g: tween_easing
                .interpolate(self.g as f32, other.g as f32, a)
                .clamp(0.0, 100.0) as i32,
            b: tween_easing
                .interpolate(self.b as f32, other.b as f32, a)
                .clamp(0.0, 100.0) as i32
        }
    }
}

#[derive(Debug)]
pub struct ColorTintAnimationTrack {
    pub slot_id: usize,
    pub regions: Vec<SamplingRegion<TintSample>>
}

#[derive(Debug)]
pub struct AnimationData {
    pub name: String,
    pub duration_in_ticks: usize,
    pub play_times: usize,
    pub rotation_tracks: Vec<AnimationTrack<RotationSample>>,
    pub transition_tracks: Vec<AnimationTrack<TransitionSample>>,
    pub scaling_tracks: Vec<AnimationTrack<ScalingSample>>,
    pub display_id_animation_tracks: Vec<DisplayIdAnimationTrack>,
    pub color_tint_animation_tracks: Vec<ColorTintAnimationTrack>,
    pub draw_order_animation_track: DrawOrderAnimationTrack
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

impl<'a> BonesMut<'a> {
    pub fn set_bone_world_rotation(&mut self, bone_id: usize, theta: f32) {
        let mut bone = &self.armature.bones[bone_id];
        let mut rotation = theta;
        if bone.inherit_rotation {
            while let Some(pid) = bone.parent_id {
                bone = &self.armature.bones[pid];
                rotation -= bone.transform.rotation;
            }
        }
        self[bone_id].transform.rotation = rotation;
    }
}

impl<'a> BonesMut<'a> {
    pub fn get_bone_world_position(
        &self,
        bone_id: usize,
        position_x: f32,
        position_y: f32,
        scale: f32,
        x_flip: DrawFlip
    ) -> (f32, f32) {
        self.armature.get_bone_world_position(bone_id, position_x, position_y, scale, x_flip)
    }

    pub fn get_bone_world_orientation(
        &self,
        bone_id: usize,
        x_flip: DrawFlip
    ) -> (f32, f32) {
        self.armature.get_bone_world_orientation(bone_id, x_flip)
    }
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

#[derive(Copy, Clone, Debug, Deserialize, PartialEq)]
pub enum Cap {
    CappedBy(f32),
    Uncapped
}
impl std::cmp::PartialOrd for Cap {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (*self, *other) {
            (Cap::Uncapped, Cap::Uncapped) => Some(Ordering::Less),
            (Cap::Uncapped, _) => Some(Ordering::Greater),
            (_, Cap::Uncapped) => Some(Ordering::Less),
            (Cap::CappedBy(l), Cap::CappedBy(r)) => l.partial_cmp(&r)
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum AnimationAction<TParam, TOutputState>
where
    TParam: Hash + Eq + Copy + Clone + Send + Sync + Sized,
    TOutputState: Hash + Eq + Copy + Clone + Send + Sync + Sized + Into<&'static str>
{
    Play(TOutputState),
    PlayBySelector{
        inner_fade_out_in_frames: usize,
        param: TParam,
        cases: Vec<(TOutputState, Cap)>
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnimationState<TParam, TOutputState>
where
    TParam: Hash + Eq + Copy + Clone + Send + Sync + Sized,
    TOutputState: Hash + Eq + Copy + Clone + Send + Sync + Sized + Into<&'static str>
{
    pub fade_out_in_frames: usize,
    pub action: AnimationAction<TParam, TOutputState>
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnimationStateMachineConfig<TParam, TInputState, TOutputState>
where
    TParam: Hash + Eq + Copy + Clone + Send + Sync + Sized,
    TInputState: Hash + Eq + Copy + Clone + Send + Sync + Sized,
    TOutputState: Hash + Eq + Copy + Clone + Send + Sync + Sized + Into<&'static str>
{
    pub start_state: TInputState,
    pub params: HashMap<TParam, f32>,
    pub states: HashMap<TInputState, AnimationState<TParam, TOutputState>>
}

pub struct AnimationStateMachine<TParam, TInputState, TOutputState>
where
    TParam: Hash + Eq + Copy + Clone + Send + Sync + Sized,
    TInputState: Hash + Eq + Copy + Clone + Send + Sync + Sized,
    TOutputState: Hash + Eq + Copy + Clone + Send + Sync + Sized + Into<&'static str>
{
    config: AnimationStateMachineConfig<TParam, TInputState, TOutputState>,
    input_state: TInputState,
    output_state: TOutputState,
}

impl<TParam, TInputState, TOutputState> AnimationStateMachine<TParam, TInputState, TOutputState>
where
    TParam: Hash + Eq + Copy + Clone + Send + Sync + Sized,
    TInputState: Hash + Eq + Copy + Clone + Send + Sync + Sized,
    TOutputState: Hash + Eq + Copy + Clone + Send + Sync + Sized + Into<&'static str>
{
    pub fn new(config: AnimationStateMachineConfig<TParam, TInputState, TOutputState>) -> Option<Self> {
        let mut config = config;
        for state in config.states.iter_mut() {
            match &mut state.1.action {
                AnimationAction::Play(_) => {}
                AnimationAction::PlayBySelector { cases, .. } => {
                    cases.sort_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Less));
                }
            }
        }
        let input_state = config.start_state;
        let state = config.states.get(&input_state)?;
        let output_state = match &state.action {
            AnimationAction::Play(value) => *value,
            AnimationAction::PlayBySelector { param, cases, .. } => {
                let param_value = config.params.get(param)?;
                let found_capped = cases
                    .iter()
                    .find(|&it| match it.1 {
                        Cap::CappedBy(x) => x >= *param_value,
                        Cap::Uncapped => false
                    });
                if let Some(capped) = found_capped {
                    capped.0
                } else {
                    let uncapped = cases
                        .iter()
                        .find(|it| if let Cap::Uncapped = it.1 {true} else {false})?;
                    uncapped.0
                }
            }
        };
        Some(Self {
            config,
            input_state,
            output_state
        })
    }
    pub fn set_input_state(&mut self, state: TInputState) {
        self.input_state = state;
    }
    pub fn set_parameter(&mut self, param: TParam, value: f32) {
        match self.config.params.get_mut(&param) {
            None => {}
            Some(x) => *x = value
        }
    }
    pub fn actualize(&mut self, armature: &mut RuntimeArmature) -> Option<()> {
        let state = self.config.states.get(&self.input_state)?;
        let (fade_out, output_state) = match &state.action {
            AnimationAction::Play(value) => (state.fade_out_in_frames, *value),
            AnimationAction::PlayBySelector { inner_fade_out_in_frames, param, cases,  } => {
                let param_value = self.config.params.get(param)?;
                let found_capped = cases
                    .iter()
                    .find(|&it| match it.1 {
                        Cap::CappedBy(x) => x >= *param_value,
                        Cap::Uncapped => false
                    });
                if let Some(capped) = found_capped {
                    ( *inner_fade_out_in_frames, capped.0 )
                } else {
                    let uncapped = cases
                        .iter()
                        .find(|it| if let Cap::Uncapped = it.1 {true} else {false})?;
                    ( *inner_fade_out_in_frames, uncapped.0 )
                }
            }
        };
        if self.output_state == output_state {
            None
        } else {
            self.output_state = output_state;
            armature.goto_and_play(self.output_state.into(), Some(fade_out));
            Some(())
        }
    }
}

fn load_texture(ctx: &mut Context, texture_bytes: &[u8]) -> macroquad::texture::Texture2D {
    let img = image::load_from_memory(texture_bytes)
        .unwrap_or_else(|e| panic!("{}", e))
        .to_rgba8();
    let (img_width, img_height) = (img.width(), img.height());
    let mut raw_bytes = img.into_raw();
    for _ in 0..3 {
        for j in 0..(img_height as usize) {
            let stride = 4 * j * img_width as usize;
            for i in (0..4 * (img_width as usize)).step_by(4) {
                if raw_bytes[stride + i] == 0 &&
                    raw_bytes[stride + i + 1] == 0 &&
                    raw_bytes[stride + i + 2] == 0 &&
                    raw_bytes[stride + i + 3] == 0 {
                    if j > 0 {
                        let prev_stride = stride - (img_width * 4) as usize;
                        if raw_bytes[prev_stride + i] != 0 ||
                            raw_bytes[prev_stride + i + 1] != 0 ||
                            raw_bytes[prev_stride + i + 2] != 0 ||
                            raw_bytes[prev_stride + i + 3] != 0 {
                            raw_bytes[stride + i] = raw_bytes[prev_stride + i];
                            raw_bytes[stride + i + 1] = raw_bytes[prev_stride + i + 1];
                            raw_bytes[stride + i + 2] = raw_bytes[prev_stride + i + 2];
                            continue;
                        }
                    }
                    if j < (img_height - 1) as usize {
                        let next_stride = stride + (img_width * 4) as usize;
                        if raw_bytes[next_stride + i] != 0 ||
                            raw_bytes[next_stride + i + 1] != 0 ||
                            raw_bytes[next_stride + i + 2] != 0 ||
                            raw_bytes[next_stride + i + 3] != 0 {
                            raw_bytes[stride + i] = raw_bytes[next_stride + i];
                            raw_bytes[stride + i + 1] = raw_bytes[next_stride + i + 1];
                            raw_bytes[stride + i + 2] = raw_bytes[next_stride + i + 2];
                            continue;
                        }
                    }
                    if i > 0 && (raw_bytes[stride + i - 1] != 0 ||
                        raw_bytes[stride + i - 2] != 0 ||
                        raw_bytes[stride + i - 3] != 0 ||
                        raw_bytes[stride + i - 4] != 0) {
                        raw_bytes[stride + i] = raw_bytes[stride + i - 4];
                        raw_bytes[stride + i + 1] = raw_bytes[stride + i - 3];
                        raw_bytes[stride + i + 2] = raw_bytes[stride + i - 2];
                        continue;
                    }
                    if i < 4 * (img_width as usize) - 4 && (raw_bytes[stride + i + 4] != 0 ||
                        raw_bytes[stride + i + 5] != 0 ||
                        raw_bytes[stride + i + 6] != 0 ||
                        raw_bytes[stride + i + 7] != 0) {
                        raw_bytes[stride + i] = raw_bytes[stride + i + 4];
                        raw_bytes[stride + i + 1] = raw_bytes[stride + i + 5];
                        raw_bytes[stride + i + 2] = raw_bytes[stride + i + 6];
                        continue;
                    }
                }
            }
        }
    }

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
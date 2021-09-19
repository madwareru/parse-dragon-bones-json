use macroquad::prelude::*;
use parse_dragon_bones_json::runtime::*;
use macroquad::miniquad::KeyCode;
use std::collections::{VecDeque};
use serde::Deserialize;
use ron::de::from_reader;

const PLAYER_RUN_SPEED: f32 = 465.0;
const BULLET_SPEED: f32 = 1000.0;
const GRAVITY: f32 = 1000.0;
const JUMP_FORCE: f32 = 800.0;

struct Bullet {
    armature: RuntimeArmature,
    position: (f32, f32),
    velocity: (f32, f32),
    lifetime: f32,
}

#[derive(Hash, Eq, PartialEq, Copy, Clone, Deserialize)]
pub enum MercenaryParam { Speed }

#[derive(Hash, Eq, PartialEq, Copy, Clone, Deserialize)]
pub enum SimpleState { Jumping, Falling, Crouching, Normal }

#[derive(Hash, Eq, PartialEq, Copy, Clone, Deserialize)]
pub enum MercenaryAnimationState {
    Jump, Fall, Crouch, CrouchBackward, Sit, Idle, Walk, WalkBackward, Run, RunBackward
}

impl Into<&'static str> for MercenaryAnimationState {
    fn into(self) -> &'static str {
        match self {
            MercenaryAnimationState::Jump => "jump",
            MercenaryAnimationState::Fall => "fall",
            MercenaryAnimationState::Crouch => "crouch",
            MercenaryAnimationState::CrouchBackward => "crouch_backward",
            MercenaryAnimationState::Sit => "sit",
            MercenaryAnimationState::Idle => "idle",
            MercenaryAnimationState::Walk => "walk",
            MercenaryAnimationState::Run => "run",
            MercenaryAnimationState::WalkBackward => "walk_backward",
            MercenaryAnimationState::RunBackward => "run_backward",
        }
    }
}

type MercenaryAnimationConfig = AnimationStateMachineConfig<
    MercenaryParam,
    SimpleState,
    MercenaryAnimationState
>;

#[macroquad::main("merc")]
async fn main() {
    let texture_bytes = include_bytes!("../src/test_assets/merc_tex.png");
    let atlas_bytes = include_bytes!("../src/test_assets/merc_tex.json");
    let skeleton_bytes = include_bytes!("../src/test_assets/merc_ske.json");
    let anim_config_bytes = include_bytes!("animation_config.ron");

    let anim_config: MercenaryAnimationConfig = from_reader(&anim_config_bytes[..]).unwrap();

    let mut animation_state_machine = AnimationStateMachine::new(anim_config).unwrap();

    let dragon_bones_data = DragonBonesData::load(skeleton_bytes, atlas_bytes, texture_bytes);
    let mut mercenary_armature = dragon_bones_data.instantiate_armature("Mercenary").unwrap();
    let mut crosshair_armature = dragon_bones_data.instantiate_armature("Crosshair").unwrap();
    let mut strange_armature = dragon_bones_data.instantiate_armature("FlipDrawOrder").unwrap();

    let gun_bone_id = mercenary_armature.get_bone_by_name("gun").unwrap();
    let head_bone_id = mercenary_armature.get_bone_by_name("Head").unwrap();
    let shoot_origin_bone_id = mercenary_armature.get_bone_by_name("shoot_origin").unwrap();

    let crosshair_bone_id = crosshair_armature.get_bone_by_name("crosshair").unwrap();

    let mut bullet_arena: Vec<Bullet> = Vec::new();
    let mut spare_bullets: VecDeque<usize> = VecDeque::new();
    let mut bullets: Vec<usize> = Vec::new();

    let mut player_x = screen_width() / 2.0;
    let mut player_y = screen_height() / 2.0;

    let mut dragon_bones_runtime = DragonBonesRuntime::new();

    let mut last_shot = 0.0;
    const FIRE_RATE: f32 = 0.15;
    const RIFLE: usize = 0;
    const SHOTGUN: usize = 1;

    let mut weapon = RIFLE;

    let mut vertical_speed = 0.0f32;

    loop {
        clear_background(Color::new(0.48, 0.46, 0.5, 1.0));

        const SCALE: f32 = 0.5;

        let (mouse_x, mouse_y) = mouse_position();

        let mut horizontal_speed = 0.0f32;
        vertical_speed += GRAVITY * get_frame_time();

        if is_key_down(KeyCode::A) {
            horizontal_speed -= PLAYER_RUN_SPEED;
        }

        if is_key_down(KeyCode::D) {
            horizontal_speed += PLAYER_RUN_SPEED;
        }

        if is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::LeftControl) {
            horizontal_speed *= 0.5;
        }

        player_x += horizontal_speed * get_frame_time();
        player_y += vertical_speed * get_frame_time();

        let mut on_ground = if player_y >= 700.0 {
            player_y = 700.0;
            vertical_speed = vertical_speed.min(0.0);
            true
        } else {
            false
        };

        if on_ground && is_key_down(KeyCode::Space) {
            vertical_speed -= JUMP_FORCE;
            on_ground = false;
        }

        let x_flip = if mouse_x > player_x {
            DrawFlip::None
        } else {
            DrawFlip::Flipped
        };

        animation_state_machine.set_parameter(
            MercenaryParam::Speed,
            match &x_flip {
                DrawFlip::None => horizontal_speed / PLAYER_RUN_SPEED,
                DrawFlip::Flipped => -horizontal_speed / PLAYER_RUN_SPEED
            },
        );

        if on_ground {
            if is_key_down(KeyCode::LeftControl) {
                animation_state_machine.set_input_state(SimpleState::Crouching);
            } else {
                animation_state_machine.set_input_state(SimpleState::Normal);
            }
        } else {
            if vertical_speed > 0.0 {
                animation_state_machine.set_input_state(SimpleState::Falling);
            } else {
                animation_state_machine.set_input_state(SimpleState::Jumping);
            }
        }

        animation_state_machine.actualize(&mut mercenary_armature);

        if is_key_pressed(KeyCode::Key1) {
            weapon = RIFLE;
        }
        if is_key_pressed(KeyCode::Key2) {
            weapon = SHOTGUN;
        }

        match weapon {
            SHOTGUN => mercenary_armature.set_slot_display_id("Gun1", Some(1)),
            _ => mercenary_armature.set_slot_display_id("Gun1", Some(0))
        }

        if is_mouse_button_down(MouseButton::Left) && (get_time() as f32 - last_shot) > FIRE_RATE {
            last_shot = get_time() as f32;
            let position = mercenary_armature.get_bone_world_position(
                shoot_origin_bone_id,
                player_x, player_y,
                SCALE,
                x_flip,
            );

            let (gun_x, gun_y) = mercenary_armature.get_bone_world_position(
                gun_bone_id,
                player_x,
                player_y,
                SCALE,
                x_flip,
            );
            let (dx, dy) = (mouse_x - gun_x, mouse_y - gun_y);
            let theta = dy.atan2(dx);

            let velocity = (BULLET_SPEED * theta.cos(), BULLET_SPEED * theta.sin());
            let bullet_id = if spare_bullets.is_empty() {
                let bullet = Bullet {
                    armature: dragon_bones_data.instantiate_armature("Bullet").unwrap(),
                    position: Default::default(),
                    velocity: Default::default(),
                    lifetime: Default::default(),
                };
                let bullet_id = bullet_arena.len();
                bullets.push(bullet_id);
                bullet_arena.push(bullet);
                bullet_id
            } else {
                let bullet_id = spare_bullets.pop_front().unwrap();
                bullets.push(bullet_id);
                bullet_id
            };
            bullet_arena[bullet_id].position = position;
            bullet_arena[bullet_id].velocity = velocity;
            bullet_arena[bullet_id].lifetime = 2.0;
            bullet_arena[bullet_id].armature.goto_and_play("fly", None);

            mercenary_armature.stack_additive_animation("fire");
            crosshair_armature.stack_additive_animation("crosshair_fire");
        }

        let dt = get_frame_time();

        mercenary_armature.update_animation_ex(dt, |bones| {
            let (bone_x, bone_y) = bones.get_bone_world_position(
                gun_bone_id,
                player_x,
                player_y,
                SCALE,
                x_flip,
            );
            let (dx, dy) = ((mouse_x - bone_x).abs(), mouse_y - bone_y);
            let theta = dy.atan2(dx);
            bones.set_bone_world_rotation(gun_bone_id, theta);
            bones.set_bone_world_rotation(head_bone_id, theta - 90.0_f32.to_radians())
        });
        mercenary_armature.draw(&mut dragon_bones_runtime, player_x, player_y, SCALE, x_flip);

        for i in (0..bullets.len()).rev() {
            if bullet_arena[bullets[i]].lifetime <= 0.0 {
                spare_bullets.push_back(bullets[i]);
                bullets.remove(i);
                continue;
            }
            let bone_id = bullet_arena[bullets[i]].armature.get_bone_by_name("Bullet").unwrap();

            let (pos_x, pos_y) = bullet_arena[bullets[i]].position;
            let (vel_x, vel_y) = bullet_arena[bullets[i]].velocity;
            let theta = vel_y.atan2(vel_x);

            bullet_arena[bullets[i]].lifetime -= dt;
            bullet_arena[bullets[i]].armature.update_animation_ex(dt, |bones| {
                bones.set_bone_world_rotation(bone_id, theta);
            });

            bullet_arena[bullets[i]].armature.draw(
                &mut dragon_bones_runtime,
                pos_x, pos_y,
                SCALE,
                DrawFlip::None,
            );

            bullet_arena[bullets[i]].position = (pos_x + vel_x * dt, pos_y + vel_y * dt);
        }

        strange_armature.update_animation(dt);
        strange_armature.draw(&mut dragon_bones_runtime, 200.0, 200.0, 1.0, DrawFlip::None);

        crosshair_armature.update_animation_ex(dt, |bones| {
            bones.set_bone_world_rotation(crosshair_bone_id, -get_time() as f32);
        });
        crosshair_armature.draw(&mut dragon_bones_runtime, mouse_x, mouse_y, SCALE * 2.0, DrawFlip::None);

        next_frame().await;
    }
}

use macroquad::prelude::*;
use parse_dragon_bones_json::runtime::*;
use macroquad::miniquad::KeyCode;
use std::collections::VecDeque;

struct Bullet {
    armature: RuntimeArmature,
    position: (f32, f32),
    velocity: (f32, f32),
    lifetime: f32
}

#[macroquad::main("merc")]
async fn main() {
    let texture_bytes = include_bytes!("../src/test_assets/merc_tex.png");
    let atlas_bytes = include_bytes!("../src/test_assets/merc_tex.json");
    let skeleton_bytes = include_bytes!("../src/test_assets/merc_ske.json");

    let dragon_bones_data = DragonBonesData::load(skeleton_bytes, atlas_bytes, texture_bytes);
    let mut runtime_armature = dragon_bones_data.instantiate_armature("Mercenary").unwrap();
    let mut crosshair_armature = dragon_bones_data.instantiate_armature("Crosshair").unwrap();

    let gun_bone_id = runtime_armature.get_bone_by_name("gun").unwrap();
    let head_bone_id = runtime_armature.get_bone_by_name("Head").unwrap();
    let shoot_origin_bone_id = runtime_armature.get_bone_by_name("shoot_origin").unwrap();

    let crosshair_bone_id = crosshair_armature.get_bone_by_name("crosshair").unwrap();

    let mut bullet_arena: Vec<Bullet> = Vec::new();
    let mut spare_bullets: VecDeque<usize> = VecDeque::new();
    let mut bullets: Vec<usize> = Vec::new();

    show_mouse(false);

    let mut draw_buffer = BufferedDrawBatcher::new();

    let mut last_shot = 0.0;
    const FIRE_RATE: f32 = 0.15;

    loop {
        clear_background(Color::new(0.48, 0.46, 0.5, 1.0));

        const SCALE: f32 = 0.5;

        let screen_center_x = screen_width() / 2.0;
        let screen_center_y = screen_height() / 2.0;

        if is_key_pressed(KeyCode::Key1) {
            runtime_armature.goto_and_play("idle", Some(4));
        } else if is_key_pressed(KeyCode::Key2) {
            runtime_armature.goto_and_play("walk", Some(3));
        } else if is_key_pressed(KeyCode::Key3) {
            runtime_armature.goto_and_play("run", Some(5));
        } else if is_key_pressed(KeyCode::Key4) {
            runtime_armature.goto_and_play("sit", Some(5));
        } else if is_key_pressed(KeyCode::Key5) {
            runtime_armature.goto_and_play("jump", Some(1));
        }

        let (mouse_x, mouse_y) = mouse_position();

        let x_flip = if mouse_x > screen_center_x {
            DrawFlip::None
        } else {
            DrawFlip::Flipped
        };

        if is_mouse_button_down(MouseButton::Left) && (get_time() as f32 - last_shot) > FIRE_RATE {
            last_shot = get_time() as f32;
            let position = runtime_armature.get_bone_world_position(
                shoot_origin_bone_id,
                screen_center_x, screen_center_y,
                SCALE,
                x_flip
            );

            let (gun_x, gun_y) = runtime_armature.get_bone_world_position(
                gun_bone_id,
                screen_center_x,
                screen_center_y,
                SCALE,
                x_flip
            );
            let (dx, dy) = (mouse_x - gun_x, mouse_y - gun_y);
            let theta = dy.atan2(dx);

            let velocity = (15.0 * theta.cos(), 15.0 * theta.sin());
            if spare_bullets.is_empty() {
                let bullet = Bullet {
                    armature:  dragon_bones_data.instantiate_armature("Bullet").unwrap(),
                    position,
                    velocity,
                    lifetime: 1.0
                };
                bullets.push(bullet_arena.len());
                bullet_arena.push(bullet);
            } else {
                let bullet_id = spare_bullets.pop_front().unwrap();
                bullet_arena[bullet_id].position = position;
                bullet_arena[bullet_id].velocity = velocity;
                bullet_arena[bullet_id].lifetime = 1.0;
                bullets.push(bullet_id);
            }
            runtime_armature.stack_additive_animation("fire");
            crosshair_armature.stack_additive_animation("fire");
        }

        runtime_armature.update_animation_ex(get_frame_time(), |bones| {
            let (bone_x, bone_y) = bones.get_bone_world_position(
                gun_bone_id,
                screen_center_x,
                screen_center_y,
                SCALE,
                x_flip
            );
            let (dx, dy) = ((mouse_x - bone_x).abs(), mouse_y - bone_y);
            let theta = dy.atan2(dx);
            bones.set_bone_world_rotation(gun_bone_id, theta);
            bones.set_bone_world_rotation(head_bone_id, theta - 90.0_f32.to_radians())
        });
        runtime_armature.draw(&mut draw_buffer, screen_center_x, screen_center_y, SCALE, x_flip);

        crosshair_armature.update_animation_ex(get_frame_time(), |bones| {
            bones.set_bone_world_rotation(crosshair_bone_id, -get_time() as f32);
        });
        crosshair_armature.draw(&mut draw_buffer, mouse_x, mouse_y, SCALE * 2.0, DrawFlip::None);

        for i in (0..bullets.len()).rev() {
            if bullet_arena[bullets[i]].lifetime <= 0.0 {
                spare_bullets.push_back(bullets[i]);
                bullets.remove(i);
                continue;
            }

            let (pos_x, pos_y) = bullet_arena[bullets[i]].position;
            let (vel_x, vel_y) = bullet_arena[bullets[i]].velocity;
            let bone_id = bullet_arena[bullets[i]].armature.get_bone_by_name("Bullet").unwrap();
            bullet_arena[bullets[i]].lifetime -= get_frame_time();
            let theta = vel_y.atan2(vel_x);
            bullet_arena[bullets[i]].armature.update_animation_ex(get_frame_time(), |bones| {
                bones.set_bone_world_rotation(bone_id, theta);
            });
            bullet_arena[bullets[i]].armature.draw(&mut draw_buffer, pos_x, pos_y, SCALE, DrawFlip::None);
            bullet_arena[bullets[i]].position = (pos_x + vel_x, pos_y + vel_y);
        }

        next_frame().await;
    }
}

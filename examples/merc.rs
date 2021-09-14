use macroquad::prelude::*;
use parse_dragon_bones_json::runtime::*;
use macroquad::miniquad::KeyCode;

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

    let crosshair_bone_id = crosshair_armature.get_bone_by_name("crosshair").unwrap();

    show_mouse(false);

    let mut draw_buffer = BufferedDrawBatcher::new();

    loop {
        clear_background(Color::new(0.48, 0.46, 0.5, 1.0));

        const SCALE: f32 = 0.75;

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
        crosshair_armature.draw(&mut draw_buffer, mouse_x, mouse_y, SCALE, DrawFlip::None);
        next_frame().await;
    }
}

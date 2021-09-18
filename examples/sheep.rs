use macroquad::prelude::*;
use parse_dragon_bones_json::runtime::*;
use macroquad::miniquad::KeyCode;

#[macroquad::main("sheep")]
async fn main() {
    let texture_bytes = include_bytes!("../src/test_assets/sheep_tex.png");
    let atlas_bytes = include_bytes!("../src/test_assets/sheep_tex.json");
    let skeleton_bytes = include_bytes!("../src/test_assets/sheep_ske.json");

    let dragon_bones_data = DragonBonesData::load(skeleton_bytes, atlas_bytes, texture_bytes);
    let mut runtime_armature = dragon_bones_data.instantiate_armature("Armature").unwrap();

    let mut runtime = DragonBonesRuntime::new();

    loop {
        clear_background(Color::new(0.48, 0.46, 0.5, 1.0));

        const SCALE: f32 = 0.9;

        let screen_center_x = screen_width() / 2.0;
        let screen_center_y = screen_height() / 2.0;

        if is_key_pressed(KeyCode::Key1) {
            runtime_armature.goto_and_play("goat_idle_anim", Some(4));
        } else if is_key_pressed(KeyCode::Key2) {
            runtime_armature.goto_and_play("goat_eat_anim", Some(4));
        } else if is_key_pressed(KeyCode::Key3) {
            runtime_armature.goto_and_play("goat_sleep_idle_anim", Some(4));
        } else if is_key_pressed(KeyCode::Key4) {
            runtime_armature.goto_and_play("goat_walk_anim", Some(4));
        } else if is_key_pressed(KeyCode::Key5) {
            runtime_armature.goto_and_play("goat_trot_anim", Some(4));
        }

        runtime_armature.update_animation(get_frame_time());

        runtime_armature.draw(&mut runtime, screen_center_x, screen_center_y, SCALE, DrawFlip::None);
        next_frame().await;
    }
}

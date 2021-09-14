use macroquad::prelude::*;
use parse_dragon_bones_json::runtime::*;
use macroquad::miniquad::KeyCode;

#[macroquad::main("robot")]
async fn main() {
    //todo: figure out what is wrong with it and fix!

    let texture_bytes = include_bytes!("../src/test_assets/robot_tex.png");
    let atlas_bytes = include_bytes!("../src/test_assets/robot_tex.json");
    let skeleton_bytes = include_bytes!("../src/test_assets/robot_ske.json");

    let dragon_bones_data = DragonBonesData::load(skeleton_bytes, atlas_bytes, texture_bytes);
    let mut runtime_armature = dragon_bones_data.instantiate_armature("Bicycle").unwrap();

    let mut draw_buffer = BufferedDrawBatcher::new();

    loop {
        clear_background(Color::new(0.48, 0.46, 0.5, 1.0));

        const SCALE: f32 = 0.3;

        let screen_center_x = screen_width() / 2.0;
        let screen_center_y = screen_height() / 2.0;

        runtime_armature.update_animation(get_frame_time());

        runtime_armature.draw(&mut draw_buffer, screen_center_x, screen_center_y, SCALE, DrawFlip::None);
        next_frame().await;
    }
}

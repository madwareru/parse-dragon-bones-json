use macroquad::prelude::*;
use parse_dragon_bones_json::runtime::*;

#[macroquad::main("draw skeleton")]
async fn main() {
    let texture_bytes = include_bytes!("../src/test_assets/rooster_tex.png");
    let atlas_bytes = include_bytes!("../src/test_assets/rooster_tex.json");
    let skeleton_bytes = include_bytes!("../src/test_assets/rooster_ske.json");

    let dragon_bones_data = DragonBonesData::load(skeleton_bytes, atlas_bytes, texture_bytes);
    let mut runtime_armature = dragon_bones_data.instantiate_armature("armatureName").unwrap();

    let mut draw_buffer = BufferedDrawBatcher::new();

    loop {
        clear_background(Color::new(0.01, 0.0, 0.05, 1.0));

        const SCALE: f32 = 0.9;

        let screen_center_x = screen_width() / 2.0;
        let screen_center_y = screen_height() / 2.0;

        let tail_bone = runtime_armature.get_bone_by_name("tail_bone");

        runtime_armature.update_animation(get_frame_time(), |bones| {
            if let Some(tail_bone) = tail_bone {
                bones[tail_bone].transform.rotation = (get_time() * 1.7).cos() as f32 * 0.1;
            }
        });
        runtime_armature.draw(&mut draw_buffer, screen_center_x, screen_center_y, SCALE);
        runtime_armature.draw_bones(screen_center_x, screen_center_y, SCALE);
        next_frame().await;
    }
}

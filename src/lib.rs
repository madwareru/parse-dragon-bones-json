pub mod skeleton_data;
pub mod atlas_data;
pub mod shared_types;
pub mod runtime;

#[cfg(test)]
mod tests {
    use crate::skeleton_data::RawSkeletonData;
    use crate::atlas_data::Atlas;

    #[test]
    fn test_deserialization_of_simple_meshy_file() {
        let bytes = include_bytes!("test_assets/test_ske.json");
        let skeleton_data: RawSkeletonData = serde_json::from_slice(bytes).unwrap();
        println!("{:?}", skeleton_data);
        let bytes = include_bytes!("test_assets/a_guy_ske.json");
        let skeleton_data: RawSkeletonData = serde_json::from_slice(bytes).unwrap();
        println!("{:?}", skeleton_data);
    }

    #[test]
    fn test_deserialization_of_composite_file() {
        let bytes = include_bytes!("test_assets/Spider_ske.json");
        let skeleton_data: RawSkeletonData = serde_json::from_slice(bytes).unwrap();
        println!("{:?}", skeleton_data);
    }

    #[test]
    fn test_deserialization_of_atlases() {
        let test_atlas = Atlas::parse(include_bytes!("test_assets/test_tex.json")).unwrap();
        println!("{:?}", test_atlas);
        let a_guy_atlas = Atlas::parse(include_bytes!("test_assets/a_guy_tex.json")).unwrap();
        println!("{:?}", a_guy_atlas);
        let spider_atlas = Atlas::parse(include_bytes!("test_assets/Spider_tex.json")).unwrap();
        println!("{:?}", spider_atlas);
    }
}

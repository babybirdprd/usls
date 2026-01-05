use anyhow::Result;
use ort::{session::Session, value::Value};
use std::path::PathBuf;

fn main() -> Result<()> {
    ort::init().commit()?;

    let models = [
        "usls/models--ResembleAI--chatterbox-turbo-ONNX/snapshots/d21799bd0354adb85e348b8a0442a8405110a2cf/onnx/speech_encoder.onnx",
        "usls/models--ResembleAI--chatterbox-turbo-ONNX/snapshots/d21799bd0354adb85e348b8a0442a8405110a2cf/onnx/embed_tokens.onnx",
        "usls/models--ResembleAI--chatterbox-turbo-ONNX/snapshots/d21799bd0354adb85e348b8a0442a8405110a2cf/onnx/language_model.onnx",
        "usls/models--ResembleAI--chatterbox-turbo-ONNX/snapshots/d21799bd0354adb85e348b8a0442a8405110a2cf/onnx/conditional_decoder.onnx",
    ];

    let local_app_data = std::env::var("LOCALAPPDATA").expect("LOCALAPPDATA not found");
    let cache_dir = std::path::PathBuf::from(local_app_data);

    for model_subpath in models {
        let path = cache_dir.join(model_subpath);
        if !path.exists() {
            println!("Path not found: {:?}", path);
            continue;
        }

        println!("\n--- Model: {:?} ---", path.file_name().unwrap());
        let session = Session::builder()?.commit_from_file(&path)?;

        println!("Inputs:");
        for input in &session.inputs {
            println!("  - {}: {:?}", input.name, input.input_type);
        }

        println!("Outputs:");
        for output in &session.outputs {
            println!("  - {}: {:?}", output.name, output.output_type);
        }
    }

    Ok(())
}

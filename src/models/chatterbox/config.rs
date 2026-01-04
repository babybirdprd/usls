use crate::{Config, ORTConfig, ProcessorConfig};

impl Config {
    pub fn chatterbox_turbo() -> Self {
        Self {
            name: "chatterbox-turbo",
            // Speech Encoder
            encoder: ORTConfig {
                file: "chatterbox/turbo/speech_encoder.onnx".to_string(),
                ..Default::default()
            },
            // Embed Tokens
            textual_encoder: ORTConfig {
                file: "chatterbox/turbo/embed_tokens.onnx".to_string(),
                ..Default::default()
            },
            // Language Model
            textual_decoder: ORTConfig {
                file: "chatterbox/turbo/language_model.onnx".to_string(),
                ..Default::default()
            },
            // Conditional Decoder
            decoder: ORTConfig {
                file: "chatterbox/turbo/conditional_decoder.onnx".to_string(),
                ..Default::default()
            },
            processor: ProcessorConfig {
                tokenizer_file: Some("chatterbox/turbo/tokenizer.json".to_string()),
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

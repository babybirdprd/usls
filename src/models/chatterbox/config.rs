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
            // Parameters for Turbo
            num_classes: Some(16), // reuse for num_kv_heads
            num_keypoints: Some(64), // reuse for head_dim
            ..Default::default()
        }
    }

    pub fn chatterbox_base() -> Self {
        Self {
            name: "chatterbox-base",
            encoder: ORTConfig {
                file: "chatterbox/base/speech_encoder.onnx".to_string(),
                ..Default::default()
            },
            textual_encoder: ORTConfig {
                file: "chatterbox/base/embed_tokens.onnx".to_string(),
                ..Default::default()
            },
            textual_decoder: ORTConfig {
                file: "chatterbox/base/language_model.onnx".to_string(),
                ..Default::default()
            },
            decoder: ORTConfig {
                file: "chatterbox/base/conditional_decoder.onnx".to_string(),
                ..Default::default()
            },
            processor: ProcessorConfig {
                tokenizer_file: Some("chatterbox/base/tokenizer.json".to_string()),
                ..Default::default()
            },
            // Assuming same params for now, can be adjusted
            num_classes: Some(16),
            num_keypoints: Some(64),
            ..Default::default()
        }
    }

    pub fn chatterbox_multilingual() -> Self {
        Self {
            name: "chatterbox-multilingual",
            encoder: ORTConfig {
                file: "chatterbox/multilingual/speech_encoder.onnx".to_string(),
                ..Default::default()
            },
            textual_encoder: ORTConfig {
                file: "chatterbox/multilingual/embed_tokens.onnx".to_string(),
                ..Default::default()
            },
            textual_decoder: ORTConfig {
                file: "chatterbox/multilingual/language_model.onnx".to_string(),
                ..Default::default()
            },
            decoder: ORTConfig {
                file: "chatterbox/multilingual/conditional_decoder.onnx".to_string(),
                ..Default::default()
            },
            processor: ProcessorConfig {
                tokenizer_file: Some("chatterbox/multilingual/tokenizer.json".to_string()),
                ..Default::default()
            },
             // Assuming same params for now, can be adjusted
            num_classes: Some(16),
            num_keypoints: Some(64),
            ..Default::default()
        }
    }
}

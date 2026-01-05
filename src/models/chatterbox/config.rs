use crate::{Config, ORTConfig, ProcessorConfig};

impl Config {
    pub fn chatterbox_turbo() -> Self {
        // Use a helper to build the path with the correct separator for the ResembleAI hub
        fn turbo_file(name: &str) -> String {
            format!("ResembleAI/chatterbox-turbo-ONNX/onnx/{}_fp16.onnx", name)
        }

        Self {
            name: "chatterbox-turbo",
            // Speech Encoder
            encoder: ORTConfig {
                file: turbo_file("speech_encoder"),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            // Embed Tokens
            textual_encoder: ORTConfig {
                file: turbo_file("embed_tokens"),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            // Language Model
            textual_decoder: ORTConfig {
                file: turbo_file("language_model"),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            // Conditional Decoder
            decoder: ORTConfig {
                file: turbo_file("conditional_decoder"),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            processor: ProcessorConfig {
                tokenizer_file: Some("ResembleAI/chatterbox-turbo-ONNX/tokenizer.json".to_string()),
                ..Default::default()
            },
            // Parameters for Turbo
            num_classes: Some(16),   // reuse for num_kv_heads
            num_keypoints: Some(64), // reuse for head_dim
            ..Default::default()
        }
    }

    pub fn chatterbox_base() -> Self {
        Self {
            name: "chatterbox-base",
            encoder: ORTConfig {
                file: "onnx-community/chatterbox-ONNX/onnx/speech_encoder.onnx".to_string(),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            textual_encoder: ORTConfig {
                file: "onnx-community/chatterbox-ONNX/onnx/embed_tokens.onnx".to_string(),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            textual_decoder: ORTConfig {
                file: "onnx-community/chatterbox-ONNX/onnx/language_model.onnx".to_string(),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            decoder: ORTConfig {
                file: "onnx-community/chatterbox-ONNX/onnx/conditional_decoder.onnx".to_string(),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            processor: ProcessorConfig {
                tokenizer_file: Some("onnx-community/chatterbox-ONNX/tokenizer.json".to_string()),
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
                file: "onnx-community/chatterbox-multilingual-ONNX/onnx/speech_encoder.onnx"
                    .to_string(),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            textual_encoder: ORTConfig {
                file: "onnx-community/chatterbox-multilingual-ONNX/onnx/embed_tokens.onnx"
                    .to_string(),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            textual_decoder: ORTConfig {
                file: "onnx-community/chatterbox-multilingual-ONNX/onnx/language_model.onnx"
                    .to_string(),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            decoder: ORTConfig {
                file: "onnx-community/chatterbox-multilingual-ONNX/onnx/conditional_decoder.onnx"
                    .to_string(),
                external_data_file: true,
                num_dry_run: 0,
                ..Default::default()
            },
            processor: ProcessorConfig {
                tokenizer_file: Some(
                    "onnx-community/chatterbox-multilingual-ONNX/tokenizer.json".to_string(),
                ),
                ..Default::default()
            },
            // Assuming same params for now, can be adjusted
            num_classes: Some(16),
            num_keypoints: Some(64),
            ..Default::default()
        }
    }
}

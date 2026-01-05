use aksr::Builder;
use anyhow::{anyhow, Result};
use ndarray::{s, Array, ArrayD, Axis};
use std::collections::HashMap;

use crate::{Config, Engine, Processor, Xs, X};

const START_SPEECH_TOKEN: i64 = 6561;
const STOP_SPEECH_TOKEN: i64 = 6562;
const SILENCE_TOKEN: i64 = 4299;

/// Chatterbox model variants.
#[derive(Debug, Copy, Clone)]
pub enum ChatterboxKind {
    /// The standard Chatterbox model.
    Base,
    /// Chatterbox Multilingual.
    Multilingual,
    /// Chatterbox Turbo (more efficient).
    Turbo,
}

#[derive(Debug, Builder)]
pub struct Chatterbox {
    /// Speech encoder for audio prompts.
    pub speech_encoder: Engine,
    /// Token embeddings model.
    pub embed_tokens: Engine,
    /// Language model (causal).
    pub language_model: Engine,
    /// Conditional decoder for audio generation.
    pub conditional_decoder: Engine,
    /// Processor for tokenization.
    pub processor: Processor,
    /// Kind of model.
    pub kind: ChatterboxKind,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Whether to run in sequential mode.
    pub sequential: bool,
}

impl Chatterbox {
    pub fn new(config: Config) -> Result<Self> {
        let speech_encoder = Engine::try_from_config(&config.encoder)?;
        let embed_tokens = Engine::try_from_config(&config.textual_encoder)?;
        let language_model = Engine::try_from_config(&config.textual_decoder)?;
        let conditional_decoder = Engine::try_from_config(&config.decoder)?;
        let processor = Processor::try_from_config(&config.processor)?;

        let kind = match config.name {
            "chatterbox-turbo" => ChatterboxKind::Turbo,
            "chatterbox-multilingual" => ChatterboxKind::Multilingual,
            _ => ChatterboxKind::Base,
        };

        // Use reused fields or defaults
        let num_kv_heads = config.num_classes.unwrap_or(16);
        let head_dim = config.num_keypoints.unwrap_or(64);

        Ok(Self {
            speech_encoder,
            embed_tokens,
            language_model,
            conditional_decoder,
            processor,
            kind,
            num_kv_heads,
            head_dim,
            sequential: config.sequential,
        })
    }

    /// Run inference to generate audio from text and a voice prompt.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to speak, including paralinguistic tags like `[laugh]`.
    /// * `audio_prompt` - PCM audio data for voice cloning (f32, expected 24kHz).
    /// * `max_new_tokens` - Maximum number of tokens to generate.
    /// * `repetition_penalty` - Penalty for repetition (e.g., 1.2).
    ///
    /// # Returns
    ///
    /// * `Vec<f32>` - Generated audio samples at 24kHz.
    pub fn run(
        &mut self,
        text: &str,
        audio_prompt: &[f32],
        max_new_tokens: usize,
        repetition_penalty: f32,
    ) -> Result<Vec<f32>> {
        // 1. Prepare Audio Input
        let audio_values =
            Array::from_shape_vec((1, audio_prompt.len()), audio_prompt.to_vec())?.into_dyn();

        // 2. Tokenize Text
        let encoding = self.processor.encode_text(text, true)?;
        let input_ids_vec: Vec<f32> = encoding.get_ids().iter().map(|&x| x as f32).collect();
        let input_ids = Array::from_shape_vec((1, input_ids_vec.len()), input_ids_vec)?.into_dyn();

        // 3. Run Speech Encoder
        if self.sequential {
            self.speech_encoder.load()?;
        }
        let encoder_outputs = self
            .speech_encoder
            .run(Xs::from(vec![X::from(audio_values)]))?;

        if self.sequential {
            self.speech_encoder.unload()?;
            self.embed_tokens.load()?;
        }

        let cond_emb = encoder_outputs["audio_features"].0.clone();
        let prompt_token = encoder_outputs["audio_tokens"].0.clone();
        let speaker_embeddings = encoder_outputs["speaker_embeddings"].clone();
        let speaker_features = encoder_outputs["speaker_features"].clone();

        // Diagnostic: check prompt tokens
        let pt_min = prompt_token.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let pt_max = prompt_token
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        tracing::info!(
            "Prompt tokens: shape={:?}, range=[{}, {}]",
            prompt_token.shape(),
            pt_min,
            pt_max
        );

        // 4. Run Embed Tokens
        let embed_outputs = self.embed_tokens.run(Xs::from(vec![X::from(input_ids)]))?;
        let text_embeds = &embed_outputs["inputs_embeds"];

        // Concatenate cond_emb and text_embeds
        let mut current_inputs_embeds =
            ndarray::concatenate(Axis(1), &[cond_emb.view(), text_embeds.view()])?.into_dyn();

        if self.sequential {
            // we keep embed_tokens loaded because it's used inside the loop
            self.language_model.load()?;
        }

        // 5. Generation Loop
        let batch_size = 1;
        let mut generate_tokens = vec![START_SPEECH_TOKEN];

        // Initialize KV cache
        let inames: Vec<String> = self
            .language_model
            .inames()
            .ok_or(anyhow!("Model inputs not available"))?
            .to_vec();

        let kv_names: Vec<String> = inames
            .iter()
            .filter(|n| n.contains("past_key_values"))
            .cloned()
            .collect();

        let mut past_key_values: HashMap<String, ArrayD<f32>> = HashMap::new();
        for name in &kv_names {
            // [batch_size, num_kv_heads, 0, head_dim]
            let shape = vec![batch_size, self.num_kv_heads, 0, self.head_dim];
            past_key_values.insert(name.clone(), Array::zeros(shape).into_dyn());
        }

        let mut attention_mask =
            Array::from_elem((batch_size, current_inputs_embeds.shape()[1]), 1.0f32).into_dyn();
        let mut position_ids = Array::from_shape_vec(
            (batch_size, current_inputs_embeds.shape()[1]),
            (0..current_inputs_embeds.shape()[1] as i64)
                .map(|x| x as f32)
                .collect(),
        )?
        .into_dyn();

        for i in 0..max_new_tokens {
            // Prepare inputs for LM
            // Order must match `inames`
            let mut lm_inputs = Vec::new();
            for name in &inames {
                if name == "inputs_embeds" {
                    lm_inputs.push(X::from(current_inputs_embeds.clone()));
                } else if name == "attention_mask" {
                    lm_inputs.push(X::from(attention_mask.clone()));
                } else if name == "position_ids" {
                    lm_inputs.push(X::from(position_ids.clone()));
                } else if let Some(kv) = past_key_values.get(name) {
                    lm_inputs.push(X::from(kv.clone()));
                } else {
                    return Err(anyhow!("Unknown input name: {}", name));
                }
            }

            let lm_outputs = self.language_model.run(Xs::from(lm_inputs))?;
            let logits = &lm_outputs["logits"];

            // Update KV cache
            for name in &kv_names {
                let output_name = name.replace("past_key_values", "present");
                if let Some(kv_out) = lm_outputs.get(&output_name) {
                    past_key_values.insert(name.clone(), kv_out.0.clone());
                }
            }

            // Slicing to get last token's logits
            let shape = logits.shape();
            let last_logits = if shape.len() == 3 {
                logits.slice(s![0, -1, ..]).into_owned().into_dyn()
            } else if shape.len() == 2 {
                logits.slice(s![-1, ..]).into_owned().into_dyn()
            } else {
                return Err(anyhow!("Unexpected logits shape: {:?}", shape));
            };

            let mut next_token_logits = last_logits.clone();
            apply_repetition_penalty(&mut next_token_logits, &generate_tokens, repetition_penalty);

            let next_token_id = argmax(&next_token_logits);
            generate_tokens.push(next_token_id);

            if i % 50 == 0 || i < 5 {
                tracing::info!(
                    "Step {}: generated token {}, logits shape={:?}",
                    i,
                    next_token_id,
                    shape
                );
            }

            if next_token_id == STOP_SPEECH_TOKEN {
                break;
            }

            // Update for next iteration
            // 1. attention_mask
            let new_mask_part = Array::ones((batch_size, 1)).into_dyn();
            attention_mask =
                ndarray::concatenate(Axis(1), &[attention_mask.view(), new_mask_part.view()])?
                    .into_dyn();

            // 2. position_ids
            let last_pos_val = *position_ids
                .iter()
                .last()
                .ok_or(anyhow!("Position IDs empty"))?;
            position_ids = Array::from_elem((batch_size, 1), last_pos_val + 1.0).into_dyn();

            // 3. input_ids for next token
            let next_token_tensor =
                Array::from_elem((batch_size, 1), next_token_id as f32).into_dyn();
            let embed_out_next = self
                .embed_tokens
                .run(Xs::from(vec![X::from(next_token_tensor)]))?;
            current_inputs_embeds = embed_out_next["inputs_embeds"].0.clone();
        }

        if self.sequential {
            self.embed_tokens.unload()?;
            self.language_model.unload()?;
        }

        // 6. Decode Audio
        let len = generate_tokens.len();
        let speech_tokens = if len > 1 { &generate_tokens[1..] } else { &[] };
        let mut speech_tokens_vec: Vec<i64> = speech_tokens.to_vec();
        if speech_tokens_vec.last() == Some(&STOP_SPEECH_TOKEN) {
            speech_tokens_vec.pop();
        }

        // Add silence
        for _ in 0..3 {
            speech_tokens_vec.push(SILENCE_TOKEN);
        }

        let mut final_tokens: Vec<f32> = prompt_token.iter().map(|&x| x as f32).collect();
        final_tokens.extend(speech_tokens_vec.iter().map(|&x| x as f32));

        let gen_tokens_arr =
            Array::from_shape_vec((1, final_tokens.len()), final_tokens)?.into_dyn();

        // Diagnostic: check generated tokens
        let gt_min = speech_tokens_vec.iter().fold(i64::MAX, |a, &b| a.min(b));
        let gt_max = speech_tokens_vec.iter().fold(i64::MIN, |a, &b| a.max(b));
        tracing::info!(
            "Generated tokens: len={}, range=[{}, {}]",
            speech_tokens_vec.len(),
            gt_min,
            gt_max
        );

        let full_speech_tokens = gen_tokens_arr;

        let dec_inames: Vec<String> = self
            .conditional_decoder
            .inames()
            .ok_or(anyhow!("Decoder inames missing"))?
            .to_vec();

        let mut ordered_inputs = Vec::new();
        for name in dec_inames {
            if name == "speech_tokens" {
                ordered_inputs.push(X::from(full_speech_tokens.clone()));
            } else if name == "speaker_embeddings" {
                ordered_inputs.push(speaker_embeddings.clone());
            } else if name == "speaker_features" {
                ordered_inputs.push(speaker_features.clone());
            } else {
                return Err(anyhow!("Unknown decoder input: {}", name));
            }
        }

        if self.sequential {
            self.conditional_decoder.load()?;
        }

        let wav_output = self.conditional_decoder.run(Xs::from(ordered_inputs))?;

        if self.sequential {
            self.conditional_decoder.unload()?;
        }

        let wav = &wav_output["waveform"]; // The probe said the output name is "waveform"
        Ok(wav.iter().cloned().collect())
    }
}

fn apply_repetition_penalty(logits: &mut ArrayD<f32>, input_ids: &[i64], penalty: f32) {
    let mut row = logits.view_mut();
    for &id in input_ids {
        if id >= 0 && (id as usize) < row.len() {
            let score = row[id as usize];
            if score < 0.0 {
                row[id as usize] = score * penalty;
            } else {
                row[id as usize] = score / penalty;
            }
        }
    }
}

fn argmax(logits: &ArrayD<f32>) -> i64 {
    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx = 0;

    for (i, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx as i64
}

use aksr::Builder;
use anyhow::{anyhow, Result};
use ndarray::{s, Array, ArrayD, Axis, IxDyn};
use std::collections::HashMap;

use crate::{Config, Engine, Processor, Xs, X};

const SAMPLE_RATE: u32 = 24000;
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
        let encoding = self.processor.encode_text(text, false)?;
        let input_ids_vec: Vec<f32> = encoding.get_ids().iter().map(|&x| x as f32).collect(); // f32 for Engine
        let input_ids = Array::from_shape_vec((1, input_ids_vec.len()), input_ids_vec)?.into_dyn();

        // 3. Run Speech Encoder
        let encoder_outputs = self
            .speech_encoder
            .run(Xs::from(vec![X::from(audio_values)]))?;
        let cond_emb = &encoder_outputs[0];
        let prompt_token = &encoder_outputs[1];
        let speaker_embeddings = &encoder_outputs[2];
        let speaker_features = &encoder_outputs[3];

        // 4. Run Embed Tokens
        let embed_outputs = self
            .embed_tokens
            .run(Xs::from(vec![X::from(input_ids.clone())]))?;
        let text_embeds = &embed_outputs[0];

        // Concatenate cond_emb and text_embeds
        let inputs_embeds =
            ndarray::concatenate(Axis(1), &[cond_emb.view(), text_embeds.view()])?.into_dyn();

        // 5. Generation Loop
        let batch_size = 1;
        let mut generate_tokens = vec![START_SPEECH_TOKEN];

        // Initialize KV cache
        // Clone names to avoid borrowing self
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
            // Shape: [batch, NUM_KV_HEADS, 0, HEAD_DIM]
            let shape = vec![batch_size, self.num_kv_heads, 0, self.head_dim];
            past_key_values.insert(name.clone(), Array::zeros(shape).into_dyn());
        }

        let mut attention_mask: Array<f32, IxDyn> =
            Array::ones((batch_size, inputs_embeds.shape()[1])).into_dyn();
        // position_ids: f32 for X/Engine
        let mut position_ids: Array<f32, IxDyn> = Array::from_shape_vec(
            (batch_size, inputs_embeds.shape()[1]),
            (0..inputs_embeds.shape()[1] as i64)
                .map(|x| x as f32)
                .collect(),
        )?
        .into_dyn();

        let mut current_inputs_embeds = inputs_embeds.clone();

        for _ in 0..max_new_tokens {
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

            // Get output names
            let onames = self
                .language_model
                .onames()
                .ok_or(anyhow!("Model outputs not available"))?;
            let logits = &lm_outputs[0];

            // Debug: log shapes and vocab size
            tracing::debug!(
                "LM output logits shape: {:?}, vocab_size: {}",
                logits.shape(),
                logits.shape().last().unwrap_or(&0)
            );

            // Update KV cache
            let mut kv_outputs = Vec::new();
            for (i, _name) in onames.iter().enumerate() {
                if i == 0 {
                    continue;
                } // Logits
                kv_outputs.push(lm_outputs[i].clone());
            }

            if kv_outputs.len() != kv_names.len() {
                return Err(anyhow!(
                    "KV cache mismatch: inputs {} vs outputs {}",
                    kv_names.len(),
                    kv_outputs.len()
                ));
            }

            for (i, name) in kv_names.iter().enumerate() {
                past_key_values.insert(name.clone(), kv_outputs[i].0.clone());
            }

            // Logits processing
            let last_logits = logits.slice(s![.., -1, ..]).into_owned().into_dyn();

            // Apply Repetition Penalty
            let mut next_token_logits = last_logits.clone();
            apply_repetition_penalty(&mut next_token_logits, &generate_tokens, repetition_penalty);

            // Greedy decoding
            let next_token_id = argmax(&next_token_logits);

            // Debug: log generated token
            tracing::debug!(
                "Generated token: {}, total tokens: {}",
                next_token_id,
                generate_tokens.len()
            );

            generate_tokens.push(next_token_id);

            if next_token_id == STOP_SPEECH_TOKEN {
                break;
            }

            // Update for next iteration
            // 1. attention_mask
            let new_mask_part: Array<f32, IxDyn> = Array::ones((batch_size, 1)).into_dyn();
            attention_mask =
                ndarray::concatenate(Axis(1), &[attention_mask.view(), new_mask_part.view()])?
                    .into_dyn();

            // 2. position_ids
            let last_pos = position_ids
                .slice(s![.., -1])
                .mapv(|x| x + 1.0)
                .insert_axis(Axis(1));
            position_ids = last_pos.into_dyn();

            // 3. input_ids for next token
            let next_input_ids = Array::from_elem((batch_size, 1), next_token_id as f32).into_dyn();
            let embed_out_next = self
                .embed_tokens
                .run(Xs::from(vec![X::from(next_input_ids)]))?;
            current_inputs_embeds = embed_out_next[0].0.clone();
        }

        // 6. Decode Audio
        let mut speech_tokens_vec = generate_tokens;
        if speech_tokens_vec.first() == Some(&START_SPEECH_TOKEN) {
            speech_tokens_vec.remove(0);
        }
        if speech_tokens_vec.last() == Some(&STOP_SPEECH_TOKEN) {
            speech_tokens_vec.pop();
        }

        // Add silence
        for _ in 0..3 {
            speech_tokens_vec.push(SILENCE_TOKEN);
        }

        let generated_tokens_arr = Array::from_shape_vec(
            (1, speech_tokens_vec.len()),
            speech_tokens_vec.iter().map(|&x| x as f32).collect(),
        )?
        .into_dyn();

        let full_speech_tokens =
            ndarray::concatenate(Axis(1), &[prompt_token.view(), generated_tokens_arr.view()])?
                .into_dyn();

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

        let wav_output = self.conditional_decoder.run(Xs::from(ordered_inputs))?;
        let wav = &wav_output[0];

        let wav_vec = wav.iter().cloned().collect();

        Ok(wav_vec)
    }
}

fn apply_repetition_penalty(logits: &mut ArrayD<f32>, input_ids: &[i64], penalty: f32) {
    let mut logits_view = logits.view_mut();
    let mut row = logits_view.slice_mut(s![0, ..]);

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
    let row = logits.slice(s![0, ..]);
    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx = 0;

    for (i, &val) in row.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx as i64
}

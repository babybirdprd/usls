# Chatterbox Example

Chatterbox is a high-performance Text-to-Speech (TTS) pipeline consisting of several models: S3Tokenizer, VoiceEncoder, T3 LM, and HiFiGAN Vocoder.

## Quick Start

```bash
# Run with CUDA acceleration
cargo run -r -F chatterbox,hf-hub,cuda --example chatterbox -- --prompt "Hello world" --device cuda:0

# Run with sequential mode (VRAM optimized)
cargo run -r -F chatterbox,hf-hub,cuda --example chatterbox -- --prompt "Hello world" --device cuda:0 --sequential
```

## Sequential Mode

Chatterbox uses multiple large models. If you have limited VRAM (e.g. 2GB), it is recommended to use the `--sequential` flag.

In sequential mode:
- Each model is loaded from disk, warmed up, used for inference, and then immediately unloaded to free memory for the next model in the pipeline.
- **Stability**: A mandatory dry-run warmup is performed during each on-demand load to ensure cuDNN is properly initialized, avoiding `HEURISTIC_QUERY_FAILED` errors.

## CUDA Stability

If you encounter `CUDNN_FE failure` errors:
1. Ensure your `ort` configuration uses `conv_algo_search = 2` (Default).
2. Ensure models are warmed up with dummy data before the first real request (handled automatically by `usls`).

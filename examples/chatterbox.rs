use anyhow::Result;
use argh::FromArgs;
use usls::{Chatterbox, Config};

#[derive(FromArgs)]
/// Chatterbox Text-to-Speech example
struct Args {
    /// input text to speak
    #[argh(
        option,
        default = "String::from(\"Hello! This is a test of Chatterbox Turbo. [laugh]\")"
    )]
    text: String,

    /// input audio prompt (wav file, preferably 24kHz)
    #[argh(option)]
    prompt: String,

    /// output audio file
    #[argh(option, default = "String::from(\"output.wav\")")]
    output: String,

    /// device to use (e.g., "cpu", "cuda:0", "directml:0")
    #[argh(option, default = "String::from(\"cpu\")")]
    device: String,

    /// use hybrid execution mode (offload large models to CPU) to avoid OOM
    #[argh(switch)]
    hybrid: bool,

    /// run models sequentially and unload them after use to save VRAM
    #[argh(switch)]
    sequential: bool,

    /// model precision (e.g., "fp32", "fp16")
    #[argh(option, default = "String::from(\"auto\")")]
    dtype: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args: Args = argh::from_env();

    // Load audio prompt
    let mut reader = hound::WavReader::open(&args.prompt)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    // Check sample rate
    if spec.sample_rate != 24000 {
        eprintln!(
            "Warning: Prompt sample rate is {}, expected 24000. This might degrade quality.",
            spec.sample_rate
        );
    }

    // Initialize model
    let device = args.device.parse::<usls::Device>()?;
    let dtype = args.dtype.parse::<usls::DType>()?;
    let mut config = Config::chatterbox_turbo()
        .with_device_all(device)
        .with_dtype_all(dtype)
        .with_sequential(args.sequential);

    if args.hybrid {
        // Offload the largest model (language_model) and the decoder to CPU
        config.textual_decoder.device = usls::Device::Cpu(0);
        config.decoder.device = usls::Device::Cpu(0);
        println!(
            "Using Hybrid mode: Language Model and Decoder on CPU, others on {:?}.",
            device
        );
    }
    if args.sequential {
        println!("Using Sequential mode: models will be loaded/unloaded one by one.");
    }
    let mut model = Chatterbox::new(config.commit()?)?;

    // Run inference
    println!("Generating audio for: \"{}\"", args.text);
    let audio_out = model.run(&args.text, &samples, 1024, 1.2)?;

    // Save output
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&args.output, spec)?;
    for sample in audio_out {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    println!("Saved audio to {}", args.output);

    Ok(())
}

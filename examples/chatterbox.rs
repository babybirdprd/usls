use anyhow::Result;
use argh::FromArgs;
use usls::{Chatterbox, Config};

#[derive(FromArgs)]
/// Chatterbox Text-to-Speech example
struct Args {
    /// input text to speak
    #[argh(option, default = "String::from(\"Hello! This is a test of Chatterbox Turbo. [laugh]\")")]
    text: String,

    /// input audio prompt (wav file, preferably 24kHz)
    #[argh(option)]
    prompt: String,

    /// output audio file
    #[argh(option, default = "String::from(\"output.wav\")")]
    output: String,
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
    let mut model = Chatterbox::new(Config::chatterbox_turbo())?;

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

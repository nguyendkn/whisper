use std::path::PathBuf;
use std::sync::Arc;

use audio_pipeline::{EnergyVad, GateConfig, SpeechProbe};
use clap::Parser;
use stream_engine::{InferenceScheduler, Session, SessionConfig, StreamEvent};
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;
use whisper_core::{WhisperConfig, WhisperModel};

mod audio_source;

/// Test local: mic hoặc file WAV -> transcript trên terminal.
#[derive(Debug, Parser)]
#[command(name = "whisper-rt", version)]
struct Args {
    /// Model GGML/GGUF của whisper.
    #[arg(long, default_value = "models/ggml-large-v3-turbo.bin")]
    model: PathBuf,
    /// Model Silero VAD của whisper.cpp. Không có thì dùng energy VAD.
    #[arg(long)]
    vad_model: Option<PathBuf>,
    /// Mã ngôn ngữ; bỏ trống để auto-detect.
    #[arg(long, default_value = "vi")]
    language: String,
    #[arg(long, default_value_t = 4)]
    threads: i32,
    /// Số inference song song.
    #[arg(long, default_value_t = 2)]
    concurrency: usize,
    /// Đọc từ file WAV thay vì mic.
    #[arg(long)]
    file: Option<PathBuf>,
    /// Nạp file nhanh nhất có thể thay vì mô phỏng thời gian thực.
    #[arg(long)]
    no_realtime: bool,
    #[arg(long, default_value_t = false)]
    use_gpu: bool,
    /// Cửa sổ decode cho partial (giây).
    #[arg(long, default_value_t = 6.0)]
    partial_window: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()))
        .init();
    whisper_core::install_logging_hooks();

    let args = Args::parse();
    let model = Arc::new(WhisperModel::load(WhisperConfig {
        model_path: args.model.clone(),
        language: (!args.language.trim().is_empty()).then(|| args.language.clone()),
        n_threads: args.threads,
        use_gpu: args.use_gpu,
        state_pool_size: args.concurrency,
        ..WhisperConfig::default()
    })?);
    let scheduler = Arc::new(InferenceScheduler::new(model, args.concurrency));

    let (event_tx, mut event_rx) = mpsc::channel::<StreamEvent>(64);
    let mut session = Session::new(
        scheduler,
        build_probe(&args)?,
        event_tx,
        SessionConfig {
            partial_window_secs: args.partial_window,
            gate: GateConfig::default(),
            ..SessionConfig::default()
        },
    );

    // Printer giữ luôn toàn văn: kết quả `Final` về sau khi vòng đọc audio đã
    // kết thúc, nên không thể hỏi session ngay sau `finish()`.
    let printer = tokio::spawn(async move {
        let mut transcript = String::new();
        while let Some(event) = event_rx.recv().await {
            match event {
                StreamEvent::Partial(update) => {
                    println!("[partial rtf={:.2}] {}", update.rtf, update.text);
                }
                StreamEvent::Final(update) => {
                    println!("[FINAL   rtf={:.2}] {}", update.rtf, update.text);
                    transcript = update.full_text;
                }
                StreamEvent::Error { message, .. } => eprintln!("[error] {message}"),
            }
        }
        transcript
    });

    match args.file.clone() {
        Some(path) => audio_source::run_file(&mut session, &path, !args.no_realtime).await?,
        None => audio_source::run_microphone(&mut session, 64).await?,
    }
    session.finish();
    // Drop session -> hết Sender khi các task inference xong -> printer kết thúc.
    drop(session);
    let transcript = printer.await.unwrap_or_default();
    if !transcript.is_empty() {
        println!("\n--- toàn văn ---\n{transcript}");
    }
    Ok(())
}

#[cfg(feature = "vad-silero")]
fn build_probe(args: &Args) -> anyhow::Result<Box<dyn SpeechProbe>> {
    match args.vad_model.as_deref() {
        Some(path) => Ok(Box::new(audio_pipeline::SileroVad::load(
            path,
            args.threads.min(2),
            args.use_gpu,
        )?)),
        None => Ok(Box::new(EnergyVad::default())),
    }
}

#[cfg(not(feature = "vad-silero"))]
fn build_probe(args: &Args) -> anyhow::Result<Box<dyn SpeechProbe>> {
    if args.vad_model.is_some() {
        eprintln!("cảnh báo: build không bật feature vad-silero, dùng energy VAD");
    }
    Ok(Box::new(EnergyVad::default()))
}

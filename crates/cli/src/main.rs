use std::path::PathBuf;
use std::sync::Arc;

use audio_pipeline::{EnergyVad, GateConfig, GatedProbe, SpeechProbe};
use clap::Parser;
use stream_engine::{
    InferenceScheduler, Session, SessionConfig, SessionEngines, StreamEvent, ThreadBudget,
};
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;
use whisper_core::{WhisperConfig, WhisperModel};

mod audio_source;
mod bench;

/// Test local: mic hoặc file audio -> transcript trên terminal.
#[derive(Debug, Parser)]
#[command(name = "whisper-rt", version)]
struct Args {
    /// Model GGML/GGUF của whisper.
    #[arg(long, default_value = "models/ggml-large-v3-turbo.bin")]
    model: PathBuf,
    /// Model Silero VAD của whisper.cpp. Không có thì dùng energy VAD.
    #[arg(long)]
    vad_model: Option<PathBuf>,
    /// Model nhỏ chạy partial (ví dụ base) trong khi --model lo lượt chốt câu.
    /// Đây là cách dùng large-v3-turbo mà partial vẫn dưới 300 ms.
    #[arg(long)]
    partial_model: Option<PathBuf>,
    /// Số thread cho model partial. Mặc định 4 — model partial không nên chiếm hết
    /// hạn mức thread của model chính.
    #[arg(long, default_value_t = 4)]
    partial_threads: i32,
    /// Tổng thread CPU cho mọi inference. 0 = tự lấy số core - 2.
    #[arg(long, default_value_t = 0)]
    cpu_budget: usize,
    /// Ngưỡng cổng năng lượng đứng trước Silero VAD (0 = tắt cổng).
    #[arg(long, default_value_t = 0.15)]
    vad_gate: f32,
    /// Mã ngôn ngữ; bỏ trống để auto-detect.
    #[arg(long, default_value = "vi")]
    language: String,
    #[arg(long, default_value_t = 12)]
    threads: i32,
    /// Số inference song song.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
    /// Đọc từ file audio (mp3/wav/flac/ogg/m4a) thay vì mic.
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
    #[arg(long)]
    flash_attn: bool,
    /// Tắt việc thu nhỏ encoder context cho partial (để so sánh khi benchmark).
    #[arg(long)]
    no_audio_ctx_scaling: bool,
    /// Bỏ segment có độ tự tin trung bình dưới ngưỡng (0 = tắt). Đo được: câu
    /// thật 0,93–0,98; ảo giác trên khoảng lặng 0,83–0,85.
    #[arg(long, default_value_t = 0.0)]
    min_confidence: f32,

    /// Chạy benchmark trên `--file` thay vì transcribe streaming.
    #[arg(long)]
    bench: bool,
    /// Số lượt đo mỗi phép trong benchmark.
    #[arg(long, default_value_t = 3)]
    repeat: usize,
    /// Độ dài đoạn dùng làm "một lượt nói" khi benchmark (giây).
    #[arg(long, default_value_t = 20.0)]
    utterance_secs: f32,
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
        flash_attn: args.flash_attn,
        state_pool_size: args.concurrency,
        scale_partial_audio_ctx: !args.no_audio_ctx_scaling,
        min_confidence: args.min_confidence,
        ..WhisperConfig::default()
    })?);
    let budget = if args.cpu_budget > 0 {
        ThreadBudget::new(args.cpu_budget)
    } else {
        ThreadBudget::auto()
    };
    let scheduler = Arc::new(InferenceScheduler::with_budget(
        model,
        Arc::clone(&budget),
        args.concurrency,
    ));

    let partial_scheduler = match args.partial_model.as_ref() {
        Some(path) => {
            let partial = Arc::new(WhisperModel::load(WhisperConfig {
                model_path: path.clone(),
                language: (!args.language.trim().is_empty()).then(|| args.language.clone()),
                n_threads: args.partial_threads,
                use_gpu: args.use_gpu,
                state_pool_size: 1,
                scale_partial_audio_ctx: !args.no_audio_ctx_scaling,
                min_confidence: args.min_confidence,
                ..WhisperConfig::default()
            })?);
            Some(Arc::new(InferenceScheduler::with_budget(
                partial,
                Arc::clone(&budget),
                1,
            )))
        }
        None => None,
    };
    let engines = SessionEngines {
        finals: Arc::clone(&scheduler),
        partials: partial_scheduler,
    };

    if args.bench {
        let path = args
            .file
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--bench cần --file <audio>"))?;
        let pcm = audio_pipeline::decode_file_to_16k_mono(&path)?;
        let label = format!(
            "model={} threads={} concurrency={} audio_ctx_scaling={} flash_attn={}",
            args.model.file_name().unwrap_or_default().to_string_lossy(),
            args.threads,
            args.concurrency,
            !args.no_audio_ctx_scaling,
            args.flash_attn,
        );
        return bench::run(
            scheduler,
            &pcm,
            &bench::BenchOptions {
                repeats: args.repeat,
                concurrency: args.concurrency,
                partial_window_secs: args.partial_window,
                utterance_secs: args.utterance_secs,
                label,
            },
        )
        .await;
    }

    let (event_tx, mut event_rx) = mpsc::channel::<StreamEvent>(64);
    let mut session = Session::new(
        engines,
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
        Some(path) => {
            let silero: Box<dyn SpeechProbe> = Box::new(audio_pipeline::SileroVad::load(
                path,
                args.threads.min(2),
                args.use_gpu,
            )?);
            // Cổng năng lượng đứng trước Silero: khoảng lặng không phải trả tiền
            // cho một lượt inference VAD.
            Ok(Box::new(GatedProbe::new(silero, args.vad_gate)))
        }
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

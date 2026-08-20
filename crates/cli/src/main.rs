use std::path::PathBuf;
use std::sync::Arc;

use audio_pipeline::{EnergyVad, GateConfig, GatedProbe, SpeechProbe};
use clap::Parser;
use stream_engine::{
    InferenceScheduler, Session, SessionConfig, SessionEngines, SessionHandle, StreamEvent,
    ThreadBudget,
};
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;
use whisper_core::{AsrBackend, WhisperBackend, WhisperConfig, WhisperModel};
use zipformer::{ZipformerBackend, ZipformerConfig};

mod audio_source;
mod bench;
mod eval;
mod wer;

/// Test local: mic hoặc file audio -> transcript trên terminal.
#[derive(Debug, Parser)]
#[command(name = "whisper-rt", version)]
struct Args {
    /// Model GGML/GGUF của whisper, hoặc thư mục ONNX khi --engine zipformer.
    #[arg(long, default_value = "models/ggml-large-v3-turbo.bin")]
    model: PathBuf,
    /// Engine ASR: whisper | zipformer (RNN-T qua sherpa-onnx).
    #[arg(long, default_value = "whisper")]
    engine: String,
    /// Zipformer: dùng bản .int8.onnx.
    #[arg(long)]
    quantized: bool,
    /// Model Silero VAD của whisper.cpp. Không có thì dùng energy VAD.
    #[arg(long)]
    vad_model: Option<PathBuf>,
    /// Model nhỏ chạy partial (ví dụ base) trong khi --model lo lượt chốt câu.
    /// Đây là cách dùng large-v3-turbo mà partial vẫn dưới 300 ms.
    #[arg(long)]
    partial_model: Option<PathBuf>,
    /// Engine cho model partial: whisper | zipformer.
    #[arg(long, default_value = "whisper")]
    partial_engine: String,
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

    /// Decode cả file trong một lượt (không VAD, không partial) — dùng làm
    /// tham chiếu "oracle" để đo phần chất lượng mất đi do streaming.
    #[arg(long)]
    offline: bool,
    /// Chạy cả bộ eval: file TSV mỗi dòng `audio<TAB>text tham chiếu`.
    #[arg(long)]
    eval_manifest: Option<PathBuf>,
    /// In WER từng clip khi chạy eval.
    #[arg(long)]
    eval_verbose: bool,
    /// So transcript với file reference và in WER.
    #[arg(long)]
    wer: Option<PathBuf>,
    /// Beam size cho lượt Final (0/1 = greedy). Mặc định 5 theo kết quả đo.
    #[arg(long, default_value_t = 5)]
    beam_size: i32,
    /// Bật temperature fallback (bước tăng temperature khi kết quả tệ).
    #[arg(long, default_value_t = 0.0)]
    temperature_inc: f32,
    /// Mồi lượt Final bằng text đã chốt trước đó.
    #[arg(long)]
    condition_on_previous: bool,
    /// Tắt LocalAgreement-2 cho partial (hiện nguyên kết quả mỗi lượt decode).
    #[arg(long)]
    no_local_agreement: bool,

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
    let backend: Arc<dyn AsrBackend> = match args.engine.as_str() {
        "whisper" => {
            let model = Arc::new(WhisperModel::load(WhisperConfig {
                model_path: args.model.clone(),
                language: (!args.language.trim().is_empty()).then(|| args.language.clone()),
                n_threads: args.threads,
                use_gpu: args.use_gpu,
                flash_attn: args.flash_attn,
                state_pool_size: args.concurrency,
                scale_partial_audio_ctx: !args.no_audio_ctx_scaling,
                min_confidence: args.min_confidence,
                beam_size: (args.beam_size > 1).then_some(args.beam_size),
                temperature_inc: args.temperature_inc,
                token_timestamps: !args.no_local_agreement,
                ..WhisperConfig::default()
            })?);
            Arc::new(WhisperBackend::new(model, args.concurrency))
        }
        "zipformer" => Arc::new(ZipformerBackend::load(ZipformerConfig {
            dir: args.model.clone(),
            quantized: args.quantized,
            n_threads: args.threads,
        })?),
        other => anyhow::bail!("engine lạ: {other:?} (whisper | zipformer)"),
    };
    let budget = if args.cpu_budget > 0 {
        ThreadBudget::new(args.cpu_budget)
    } else {
        ThreadBudget::auto()
    };
    let scheduler = Arc::new(InferenceScheduler::with_budget(
        backend,
        Arc::clone(&budget),
    ));

    let partial_scheduler = match args.partial_model.as_ref() {
        Some(path) => {
            let partial: Arc<dyn AsrBackend> = match args.partial_engine.as_str() {
                "whisper" => Arc::new(WhisperBackend::new(
                    Arc::new(WhisperModel::load(WhisperConfig {
                        model_path: path.clone(),
                        language: (!args.language.trim().is_empty()).then(|| args.language.clone()),
                        n_threads: args.partial_threads,
                        use_gpu: args.use_gpu,
                        state_pool_size: 1,
                        scale_partial_audio_ctx: !args.no_audio_ctx_scaling,
                        min_confidence: args.min_confidence,
                        token_timestamps: !args.no_local_agreement,
                        ..WhisperConfig::default()
                    })?),
                    1,
                )),
                "zipformer" => Arc::new(ZipformerBackend::load(ZipformerConfig {
                    dir: path.clone(),
                    quantized: args.quantized,
                    n_threads: args.partial_threads,
                })?),
                other => anyhow::bail!("partial-engine lạ: {other:?}"),
            };
            Some(Arc::new(InferenceScheduler::with_budget(
                partial,
                Arc::clone(&budget),
            )))
        }
        None => None,
    };
    let engines = SessionEngines {
        finals: Arc::clone(&scheduler),
        partials: partial_scheduler,
    };

    if let Some(manifest) = args.eval_manifest.as_deref() {
        return eval::run(scheduler, manifest, args.eval_verbose).await;
    }

    if args.offline {
        let path = args
            .file
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--offline cần --file <audio>"))?;
        let pcm = audio_pipeline::decode_file_to_16k_mono(&path)?;
        let started = std::time::Instant::now();
        let result = scheduler
            .submit(pcm, whisper_core::DecodeMode::Final, None, None)
            .await?;
        let text = result.text();
        println!("{text}");
        eprintln!(
            "offline: audio_ms={} inference_ms={} rtf={:.3}",
            result.audio_ms,
            result.inference_ms,
            result.rtf()
        );
        eprintln!("wall_ms={}", started.elapsed().as_millis());
        report_wer(args.wer.as_deref(), &text)?;
        return Ok(());
    }

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
    let session = Session::new(
        engines,
        build_probe(&args)?,
        event_tx,
        SessionConfig {
            partial_window_secs: args.partial_window,
            gate: GateConfig::default(),
            condition_on_previous: args.condition_on_previous,
            local_agreement: !args.no_local_agreement,
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
                    // Phần đã chốt in trước dấu | , phần còn có thể đổi in sau.
                    let pending = update
                        .text
                        .strip_prefix(update.stable_text.as_str())
                        .unwrap_or(&update.text)
                        .trim();
                    println!(
                        "[partial rtf={:.2}] {} | {}",
                        update.rtf, update.stable_text, pending
                    );
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

    let session = SessionHandle::spawn(session);
    match args.file.clone() {
        Some(path) => audio_source::run_file(&session, &path, !args.no_realtime).await?,
        None => audio_source::run_microphone(&session, 64).await?,
    }
    // Drop handle -> task blocking finish() rồi drop Session -> hết Sender khi các
    // task inference xong -> printer kết thúc.
    drop(session);
    let transcript = printer.await.unwrap_or_default();
    if !transcript.is_empty() {
        println!("\n--- toàn văn ---\n{transcript}");
    }
    report_wer(args.wer.as_deref(), &transcript)?;
    Ok(())
}

/// In WER nếu có reference. Dùng để so các cấu hình decode với nhau bằng số.
fn report_wer(reference: Option<&std::path::Path>, hypothesis: &str) -> anyhow::Result<()> {
    let Some(path) = reference else {
        return Ok(());
    };
    let reference = std::fs::read_to_string(path)?;
    let report = wer::compare(&reference, hypothesis);
    println!(
        "WER={:.4} errors={} sub={} del={} ins={} ref_words={}",
        report.wer(),
        report.errors(),
        report.substitutions,
        report.deletions,
        report.insertions,
        report.reference_words,
    );
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

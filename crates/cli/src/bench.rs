//! Đo hiệu năng inference: độ trễ một lượt partial/final và throughput khi chạy
//! song song. In ra dòng `BENCH key=value` để script sweep parse được.
//!
//! Vì sao đo cả hai: độ trễ partial là con số người dùng cảm nhận, còn throughput
//! quyết định một máy chịu được bao nhiêu session cùng lúc.

use std::sync::Arc;
use std::time::Instant;

use stream_engine::InferenceScheduler;
use whisper_core::{DecodeMode, WHISPER_SAMPLE_RATE};

pub struct BenchOptions {
    pub repeats: usize,
    pub concurrency: usize,
    pub partial_window_secs: f32,
    pub utterance_secs: f32,
    /// Nhãn mô tả cấu hình, ghép vào mọi dòng BENCH.
    pub label: String,
}

pub async fn run(
    scheduler: Arc<InferenceScheduler>,
    pcm: &[f32],
    opts: &BenchOptions,
) -> anyhow::Result<()> {
    let utterance = take_secs(pcm, opts.utterance_secs);
    let partial = take_last_secs(&utterance, opts.partial_window_secs);

    // Warm-up: lượt đầu gánh cả cấp phát state và làm nóng cache.
    scheduler
        .submit(utterance.clone(), DecodeMode::Final, None)
        .await?;

    let partial_ms = measure(&scheduler, &partial, DecodeMode::Partial, opts.repeats).await?;
    let final_ms = measure(&scheduler, &utterance, DecodeMode::Final, opts.repeats).await?;

    report("partial", &partial_ms, secs_of(&partial), opts);
    report("final", &final_ms, secs_of(&utterance), opts);

    // Throughput: N lượt Final chạy song song, giới hạn bởi semaphore.
    let started = Instant::now();
    let mut tasks = Vec::with_capacity(opts.concurrency);
    for _ in 0..opts.concurrency {
        let scheduler = Arc::clone(&scheduler);
        let pcm = utterance.clone();
        tasks.push(tokio::spawn(async move {
            scheduler.submit(pcm, DecodeMode::Final, None).await
        }));
    }
    for task in tasks {
        task.await??;
    }
    let wall_ms = started.elapsed().as_millis() as u64;
    let audio_secs = secs_of(&utterance) * opts.concurrency as f32;
    println!(
        "BENCH kind=throughput {} streams={} wall_ms={} audio_secs={:.1} aggregate_rtf={:.4} \
         streams_at_rtf1={:.1}",
        opts.label,
        opts.concurrency,
        wall_ms,
        audio_secs,
        wall_ms as f32 / (audio_secs * 1_000.0),
        audio_secs * 1_000.0 / wall_ms as f32,
    );
    Ok(())
}

async fn measure(
    scheduler: &Arc<InferenceScheduler>,
    pcm: &[f32],
    mode: DecodeMode,
    repeats: usize,
) -> anyhow::Result<Vec<u64>> {
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats.max(1) {
        let started = Instant::now();
        scheduler.submit(pcm.to_vec(), mode, None).await?;
        samples.push(started.elapsed().as_millis() as u64);
    }
    samples.sort_unstable();
    Ok(samples)
}

fn report(kind: &str, samples: &[u64], audio_secs: f32, opts: &BenchOptions) {
    let median = samples[samples.len() / 2];
    let worst = *samples.last().unwrap_or(&0);
    println!(
        "BENCH kind={} {} audio_secs={:.1} median_ms={} worst_ms={} rtf={:.4}",
        kind,
        opts.label,
        audio_secs,
        median,
        worst,
        median as f32 / (audio_secs * 1_000.0),
    );
}

fn secs_of(pcm: &[f32]) -> f32 {
    pcm.len() as f32 / WHISPER_SAMPLE_RATE as f32
}

fn take_secs(pcm: &[f32], secs: f32) -> Vec<f32> {
    let want = (WHISPER_SAMPLE_RATE as f32 * secs) as usize;
    pcm[..want.min(pcm.len())].to_vec()
}

fn take_last_secs(pcm: &[f32], secs: f32) -> Vec<f32> {
    let want = (WHISPER_SAMPLE_RATE as f32 * secs) as usize;
    pcm[pcm.len().saturating_sub(want)..].to_vec()
}

//! Chạy cả một bộ eval (manifest TSV: `đường_dẫn_audio<TAB>text_tham_chiếu`) và in
//! WER tổng của toàn bộ corpus.
//!
//! WER tổng tính trên toàn corpus (tổng lỗi / tổng từ tham chiếu), không phải trung
//! bình WER từng clip — clip ngắn không được cân bằng clip dài.

use std::path::Path;
use std::sync::Arc;

use stream_engine::InferenceScheduler;
use whisper_core::DecodeMode;

use crate::wer;

#[derive(Debug, Default)]
struct Totals {
    clips: usize,
    failed: usize,
    ref_words: usize,
    substitutions: usize,
    deletions: usize,
    insertions: usize,
    audio_ms: u64,
    inference_ms: u64,
}

impl Totals {
    fn errors(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }

    fn wer(&self) -> f32 {
        if self.ref_words == 0 {
            return 0.0;
        }
        self.errors() as f32 / self.ref_words as f32
    }
}

pub async fn run(
    scheduler: Arc<InferenceScheduler>,
    manifest: &Path,
    verbose: bool,
) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(manifest)?;
    let mut totals = Totals::default();

    for line in text.lines() {
        let Some((path, reference)) = line.split_once('\t') else {
            continue;
        };
        let pcm = match audio_pipeline::decode_file_to_16k_mono(Path::new(path)) {
            Ok(pcm) => pcm,
            Err(err) => {
                eprintln!("bỏ {path}: {err}");
                totals.failed += 1;
                continue;
            }
        };

        let result = match scheduler.submit(pcm, DecodeMode::Final, None).await {
            Ok(result) => result,
            Err(err) => {
                eprintln!("bỏ {path}: {err}");
                totals.failed += 1;
                continue;
            }
        };
        let hypothesis = result.text();
        let report = wer::compare(reference, &hypothesis);

        totals.clips += 1;
        totals.ref_words += report.reference_words;
        totals.substitutions += report.substitutions;
        totals.deletions += report.deletions;
        totals.insertions += report.insertions;
        totals.audio_ms += result.audio_ms as u64;
        totals.inference_ms += result.inference_ms;

        if verbose {
            println!(
                "{:<28} WER={:.3} sub={} del={} ins={} ref={}",
                Path::new(path)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy(),
                report.wer(),
                report.substitutions,
                report.deletions,
                report.insertions,
                report.reference_words
            );
        }
    }

    println!(
        "EVAL clips={} failed={} ref_words={} WER={:.4} sub={} del={} ins={} rtf={:.3}",
        totals.clips,
        totals.failed,
        totals.ref_words,
        totals.wer(),
        totals.substitutions,
        totals.deletions,
        totals.insertions,
        totals.inference_ms as f32 / totals.audio_ms.max(1) as f32,
    );
    Ok(())
}

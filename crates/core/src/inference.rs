use std::time::Instant;

use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

use crate::{
    config::{WhisperConfig, WHISPER_SAMPLE_RATE},
    error::AsrError,
    model::WhisperModel,
};

/// Chế độ decode. Partial ưu tiên độ trễ, Final ưu tiên chất lượng.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMode {
    /// Cửa sổ ngắn, greedy, gộp về một segment — chạy nhiều lần mỗi giây.
    Partial,
    /// Cả câu, cho phép nhiều segment và beam search nếu config bật.
    Final,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
}

#[derive(Debug, Clone)]
pub struct TranscriptResult {
    pub segments: Vec<Segment>,
    pub mode: DecodeMode,
    /// Độ dài audio đưa vào.
    pub audio_ms: u32,
    /// Thời gian inference thực tế.
    pub inference_ms: u64,
}

impl TranscriptResult {
    pub fn text(&self) -> String {
        let mut out = String::new();
        for segment in &self.segments {
            let piece = segment.text.trim();
            if piece.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(piece);
        }
        out
    }

    /// Real-time factor: <1.0 nghĩa là transcribe nhanh hơn tốc độ nói.
    pub fn rtf(&self) -> f32 {
        if self.audio_ms == 0 {
            return f32::INFINITY;
        }
        self.inference_ms as f32 / self.audio_ms as f32
    }
}

/// Transcribe một buffer PCM f32 16 kHz mono. **Blocking** — caller phải đưa
/// vào `spawn_blocking` hoặc thread pool riêng, đây là tải CPU/GPU chứ không
/// phải async I/O.
pub fn transcribe(
    model: &WhisperModel,
    state: &mut WhisperState,
    pcm: &[f32],
    mode: DecodeMode,
) -> Result<TranscriptResult, AsrError> {
    let config = model.config();
    let audio_ms = (pcm.len() as u64 * 1_000 / WHISPER_SAMPLE_RATE as u64) as u32;
    if pcm.len() < config.min_samples() {
        return Err(AsrError::AudioTooShort {
            got_ms: audio_ms,
            min_ms: config.min_audio_ms,
        });
    }

    let params = build_params(config, mode, pcm.len());
    let started = Instant::now();
    state.full(params, pcm).map_err(AsrError::Inference)?;
    let inference_ms = started.elapsed().as_millis() as u64;

    let n_segments = state.full_n_segments();
    let mut segments = Vec::with_capacity(n_segments.max(0) as usize);
    let mut dropped = 0;
    for i in 0..n_segments {
        let Some(segment) = state.get_segment(i) else {
            continue;
        };
        // Chặn ảo giác trên khoảng lặng: whisper vẫn sinh text khi không có tiếng
        // nói, nhưng tự báo no_speech cao — dùng luôn tín hiệu đó.
        let no_speech = segment.no_speech_probability();
        let confidence = mean_token_probability(&segment);
        tracing::debug!(
            no_speech,
            confidence,
            text = segment.to_str_lossy().unwrap_or_default().as_ref(),
            "segment"
        );
        // Chặn ảo giác trên khoảng lặng theo hai tín hiệu độc lập: whisper tự báo
        // no_speech, và độ tự tin trung bình của token trong segment.
        if no_speech > config.no_speech_thold || confidence < config.min_confidence {
            dropped += 1;
            continue;
        }
        segments.push(Segment {
            text: segment
                .to_str_lossy()
                .map_err(AsrError::Inference)?
                .into_owned(),
            // whisper trả timestamp theo đơn vị 10 ms.
            start_ms: segment.start_timestamp() * 10,
            end_ms: segment.end_timestamp() * 10,
        });
    }

    let result = TranscriptResult {
        segments,
        mode,
        audio_ms,
        inference_ms,
    };
    tracing::debug!(
        ?mode,
        audio_ms,
        inference_ms,
        rtf = result.rtf(),
        n_segments,
        dropped,
        "transcribed chunk"
    );
    Ok(result)
}

/// Xác suất trung bình của các token trong segment — ảo giác thường có độ tự tin
/// thấp hơn rõ rệt so với câu nói thật.
fn mean_token_probability(segment: &whisper_rs::WhisperSegment<'_>) -> f32 {
    let n_tokens = segment.n_tokens();
    if n_tokens <= 0 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut counted = 0;
    for i in 0..n_tokens {
        if let Some(token) = segment.get_token(i) {
            total += token.token_probability();
            counted += 1;
        }
    }
    if counted == 0 {
        return 0.0;
    }
    total / counted as f32
}

fn build_params<'a>(
    config: &'a WhisperConfig,
    mode: DecodeMode,
    samples: usize,
) -> FullParams<'a, 'a> {
    let strategy = match (mode, config.beam_size) {
        (DecodeMode::Final, Some(beam_size)) => SamplingStrategy::BeamSearch {
            beam_size,
            patience: 1.0,
        },
        _ => SamplingStrategy::Greedy { best_of: 1 },
    };

    let mut params = FullParams::new(strategy);
    params.set_n_threads(config.n_threads);
    params.set_translate(config.translate);
    params.set_temperature(config.temperature);
    // Mỗi cửa sổ là độc lập: sliding window đã chồng lấn nhau, mang context
    // của lượt trước sang sẽ sinh vòng lặp lặp chữ.
    params.set_no_context(true);
    // Partial gộp một segment cho ngắn; Final phải để whisper tự cắt câu,
    // nếu không văn bản dài sẽ bị cắt cụt còn một segment.
    params.set_single_segment(mode == DecodeMode::Partial);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_print_timestamps(false);
    // Partial chỉ cần encoder chạy đúng độ dài cửa sổ; Final giữ nguyên full
    // context để không mất chất lượng ở đoạn cuối câu.
    if mode == DecodeMode::Partial {
        if let Some(audio_ctx) = config.audio_ctx_for(samples) {
            params.set_audio_ctx(audio_ctx);
        }
    }
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_no_speech_thold(config.no_speech_thold);
    if let Some(language) = config.language.as_deref() {
        params.set_language(Some(language));
    } else {
        params.set_detect_language(true);
    }
    if let Some(prompt) = config.initial_prompt.as_deref() {
        params.set_initial_prompt(prompt);
    }
    params
}

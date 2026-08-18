use std::io::Cursor;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};
use whisper_core::{DecodeMode, WHISPER_SAMPLE_RATE};

use crate::state::AppState;

pub async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "whisper_cpp": whisper_core::whisper_cpp_version(),
        "model": state.cfg.model.path,
        "max_concurrent_inference": state.scheduler.max_concurrent(),
        "available_permits": state.scheduler.available_permits(),
    }))
}

/// Batch: POST một file WAV (PCM 16-bit hoặc float, sample rate nào cũng được)
/// trong body, nhận lại toàn văn kèm segment.
pub async fn transcribe_wav(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let pcm = decode_wav(&body).map_err(|err| bad_request(&err))?;
    let result = state
        .scheduler
        .submit(pcm, DecodeMode::Final)
        .await
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": err.to_string() })),
            )
        })?;

    Ok(Json(json!({
        "text": result.text(),
        "audio_ms": result.audio_ms,
        "inference_ms": result.inference_ms,
        "rtf": result.rtf(),
        "segments": result.segments.iter().map(|segment| json!({
            "text": segment.text,
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
        })).collect::<Vec<_>>(),
    })))
}

/// Đọc WAV thành mono f32 16 kHz.
fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let reader =
        hound::WavReader::new(Cursor::new(bytes)).map_err(|e| format!("invalid wav: {e}"))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?
        }
    };

    if spec.sample_rate == WHISPER_SAMPLE_RATE && spec.channels == 1 {
        return Ok(samples);
    }

    let mut resampler =
        audio_pipeline::AudioResampler::new(spec.sample_rate, spec.channels as usize)
            .map_err(|e| e.to_string())?;
    let mut pcm = resampler.push(&samples).map_err(|e| e.to_string())?;
    pcm.extend(resampler.flush().map_err(|e| e.to_string())?);
    Ok(pcm)
}

fn bad_request(message: &str) -> (StatusCode, Json<Value>) {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": message })))
}

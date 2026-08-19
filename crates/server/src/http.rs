use audio_pipeline::decode_bytes_to_16k_mono;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};
use whisper_core::DecodeMode;

use crate::state::AppState;

pub async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "whisper_cpp": whisper_core::whisper_cpp_version(),
        "model": state.cfg.model.path,
        "cpu_thread_budget": state.scheduler.budget().total(),
        "threads_available": state.scheduler.available_permits(),
        "max_concurrent_inference": state.scheduler.max_concurrent(),
        "partial_model": state.partial_scheduler.is_some(),
        "language_models": state.language_schedulers.keys().collect::<Vec<_>>(),
    }))
}

/// Batch: POST một file audio (mp3, wav, flac, ogg, m4a — sample rate/số kênh nào
/// cũng được) trong body, nhận lại toàn văn kèm segment.
pub async fn transcribe(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let pcm =
        decode_bytes_to_16k_mono(body.to_vec()).map_err(|err| bad_request(&err.to_string()))?;
    let result = state
        .scheduler
        .submit(pcm, DecodeMode::Final, None)
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

fn bad_request(message: &str) -> (StatusCode, Json<Value>) {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": message })))
}

use audio_pipeline::decode_bytes_to_16k_mono;
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::Html;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use whisper_core::DecodeMode;

use crate::state::AppState;

/// UI transcript realtime. Nhúng thẳng vào binary để deploy chỉ cần một file, và
/// không có tài nguyên ngoài nào (CDN) — trang phải chạy được trong mạng nội bộ.
/// `no-cache`: browser phải revalidate mỗi lần — một bản JS cũ nằm trong cache là
/// một buổi debug "nói mà không thấy chữ" (đã xảy ra thật).
pub async fn index() -> ([(header::HeaderName, &'static str); 1], Html<&'static str>) {
    (
        [(header::CACHE_CONTROL, "no-cache")],
        Html(include_str!("../assets/index.html")),
    )
}

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
#[derive(Debug, Deserialize)]
pub struct TranscribeParams {
    /// `?language=vi` — bỏ trống thì auto-detect (tốn thêm một lượt decode).
    #[serde(default)]
    language: Option<String>,
}

pub async fn transcribe(
    State(state): State<AppState>,
    Query(params): Query<TranscribeParams>,
    body: Bytes,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    // Giải mã mp3/flac là CPU-bound hàng giây với file dài — không được chạy trên
    // reactor của tokio, nếu không mọi kết nối khác cùng khựng.
    let pcm = tokio::task::spawn_blocking(move || decode_bytes_to_16k_mono(body.to_vec()))
        .await
        .map_err(|err| bad_request(&err.to_string()))?
        .map_err(|err| bad_request(&err.to_string()))?;
    let result = state
        .scheduler
        .submit(pcm, DecodeMode::Final, None, params.language)
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

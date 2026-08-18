//! Giao thức streaming.
//!
//! Client mở `GET /v1/stream?sample_rate=48000&channels=1`, gửi **binary frame**
//! là PCM i16 little-endian ở sample rate đã khai báo (mặc định 16 kHz mono, khi
//! đó server không phải resample). Text frame `{"type":"eos"}` để chốt câu cuối.
//!
//! Server trả text frame JSON: `ready`, `partial`, `final`, `error`.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;

use audio_pipeline::{pcm_i16_le_to_f32, AudioResampler, TARGET_SAMPLE_RATE};
use stream_engine::{Session, StreamEvent, TranscriptUpdate};

use crate::state::AppState;

/// Frame text từ client. Hiện chỉ có một lệnh: chốt câu cuối.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMessage {
    Eos,
}

#[derive(Debug, Deserialize)]
pub struct StreamParams {
    #[serde(default = "default_sample_rate")]
    sample_rate: u32,
    #[serde(default = "default_channels")]
    channels: usize,
}

fn default_sample_rate() -> u32 {
    TARGET_SAMPLE_RATE
}

fn default_channels() -> usize {
    1
}

/// Chờ tối đa bao lâu cho các lượt inference còn dở khi client đã ngắt.
const FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

pub async fn stream_handler(
    ws: WebSocketUpgrade,
    Query(params): Query<StreamParams>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state, params))
}

async fn handle_socket(socket: WebSocket, state: AppState, params: StreamParams) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    let mut resampler = match AudioResampler::new(params.sample_rate, params.channels) {
        Ok(resampler) => resampler,
        Err(err) => {
            let _ = ws_tx
                .send(Message::text(
                    json!({ "type": "error", "message": err.to_string() }).to_string(),
                ))
                .await;
            return;
        }
    };

    let (event_tx, mut event_rx) = mpsc::channel::<StreamEvent>(64);
    let mut session = Session::new(
        state.scheduler.clone(),
        state.new_probe(),
        event_tx,
        state.cfg.session_config(),
    );
    let session_id = session.id();
    tracing::info!(
        %session_id,
        sample_rate = params.sample_rate,
        channels = params.channels,
        "streaming session opened"
    );

    // Task đẩy kết quả ra client. Tách khỏi vòng đọc audio để một bên chậm không
    // chặn bên kia.
    let forward = tokio::spawn(async move {
        let _ = ws_tx
            .send(Message::text(
                json!({ "type": "ready", "session_id": session_id }).to_string(),
            ))
            .await;
        while let Some(event) = event_rx.recv().await {
            if ws_tx.send(Message::text(encode(&event))).await.is_err() {
                break;
            }
        }
    });

    while let Some(Ok(message)) = ws_rx.next().await {
        match message {
            Message::Binary(data) => {
                let pcm = pcm_i16_le_to_f32(&data);
                match resampler.push(&pcm) {
                    Ok(pcm16k) => session.push_pcm(&pcm16k),
                    Err(err) => tracing::warn!(%session_id, %err, "resample failed"),
                }
            }
            Message::Text(text) => match serde_json::from_str::<ClientMessage>(text.as_str()) {
                Ok(ClientMessage::Eos) => {
                    if let Ok(tail) = resampler.flush() {
                        session.push_pcm(&tail);
                    }
                    session.finish();
                }
                Err(err) => tracing::debug!(%session_id, %err, "ignoring unknown text frame"),
            },
            Message::Close(_) => break,
            _ => {}
        }
    }

    if let Ok(tail) = resampler.flush() {
        session.push_pcm(&tail);
    }
    session.finish();
    // Drop session để mọi Sender còn lại chỉ nằm trong các task inference đang
    // chạy: khi chúng xong, channel đóng và `forward` tự kết thúc — không cần
    // sleep đoán chừng. Timeout để một lượt inference treo không giữ socket mãi.
    drop(session);
    if tokio::time::timeout(FLUSH_TIMEOUT, forward).await.is_err() {
        tracing::warn!(%session_id, "timed out flushing pending results");
    }
    tracing::info!(%session_id, "session closed");
}

fn encode(event: &StreamEvent) -> String {
    match event {
        StreamEvent::Partial(update) => payload("partial", update),
        StreamEvent::Final(update) => payload("final", update),
        StreamEvent::Error {
            session_id,
            message,
        } => json!({
            "type": "error",
            "session_id": session_id,
            "message": message,
        })
        .to_string(),
    }
}

fn payload(kind: &str, update: &TranscriptUpdate) -> String {
    json!({
        "type": kind,
        "session_id": update.session_id,
        "utterance": update.utterance,
        "text": update.text,
        "full_text": update.full_text,
        "audio_ms": update.audio_ms,
        "rtf": update.rtf,
    })
    .to_string()
}

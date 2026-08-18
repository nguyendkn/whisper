//! Lõi inference: nhận PCM f32 16 kHz mono, trả text. Không biết gì về mic,
//! WebSocket hay session — nhờ vậy đổi backend (whisper.cpp → Voxtral,
//! Parakeet) chỉ đụng crate này.

pub mod config;
pub mod error;
pub mod inference;
pub mod model;
pub mod state_pool;

pub use config::{WhisperConfig, WHISPER_CHUNK_SECS, WHISPER_SAMPLE_RATE};
pub use error::AsrError;
pub use inference::{transcribe, DecodeMode, Segment, TranscriptResult};
pub use model::WhisperModel;
pub use state_pool::{PooledState, StatePool};

/// Version whisper.cpp mà binary này link tới — log ra khi khởi động để biết
/// chính xác backend nào đang chạy.
pub fn whisper_cpp_version() -> &'static str {
    whisper_rs::WHISPER_CPP_VERSION
}

/// Đưa log của whisper.cpp/GGML vào `tracing` thay vì stderr.
pub fn install_logging_hooks() {
    whisper_rs::install_logging_hooks();
}

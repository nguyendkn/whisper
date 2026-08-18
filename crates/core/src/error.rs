use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum AsrError {
    #[error("model file not found: {0}")]
    ModelMissing(PathBuf),
    #[error("failed to load model {path}: {source}")]
    ModelLoad {
        path: PathBuf,
        #[source]
        source: whisper_rs::WhisperError,
    },
    #[error("failed to create whisper state: {0}")]
    StateCreate(#[source] whisper_rs::WhisperError),
    #[error("inference failed: {0}")]
    Inference(#[source] whisper_rs::WhisperError),
    /// whisper.cpp trả rỗng hoặc lỗi với audio quá ngắn — chặn từ sớm để
    /// không tốn một lượt encode 30 s cho vài chục ms audio.
    #[error("audio too short: {got_ms} ms < {min_ms} ms")]
    AudioTooShort { got_ms: u32, min_ms: u32 },
}

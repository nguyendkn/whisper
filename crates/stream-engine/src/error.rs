#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error(transparent)]
    Asr(#[from] whisper_core::AsrError),
    #[error(transparent)]
    Audio(#[from] audio_pipeline::AudioError),
    #[error("inference task failed: {0}")]
    Join(String),
    #[error("scheduler is shutting down")]
    Shutdown,
}

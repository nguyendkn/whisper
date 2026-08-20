//! Trait chung cho mọi engine ASR.
//!
//! Vì sao cần: `InferenceScheduler` từng gắn cứng với whisper.cpp; muốn chạy một
//! model kiến trúc khác (Zipformer RNN-T qua sherpa-onnx) thì mọi thứ phía trên —
//! session, LocalAgreement, budget thread, eval harness — phải dùng lại nguyên vẹn.
//! Ranh giới đúng là "PCM 16 kHz mono vào, `TranscriptResult` ra".

use std::sync::Arc;

use crate::{
    config::WhisperConfig, error::AsrError, inference::transcribe, inference::DecodeMode,
    inference::TranscriptResult, model::WhisperModel, state_pool::StatePool,
};

pub trait AsrBackend: Send + Sync {
    /// Blocking — caller đưa vào `spawn_blocking`, như `transcribe` của whisper.
    fn transcribe(
        &self,
        pcm: &[f32],
        mode: DecodeMode,
        prompt: Option<&str>,
        language: Option<&str>,
    ) -> Result<TranscriptResult, AsrError>;

    /// Số thread một lượt inference chiếm — scheduler xin đúng ngần này permit
    /// từ `ThreadBudget`.
    fn n_threads(&self) -> usize;

    /// Tên hiển thị trong log/health.
    fn name(&self) -> &'static str;
}

/// whisper.cpp: model share qua `Arc`, mỗi lượt inference mượn một `WhisperState`
/// từ pool (KV cache hàng trăm MB — không cấp phát lại mỗi chunk).
pub struct WhisperBackend {
    model: Arc<WhisperModel>,
    pool: Arc<StatePool>,
}

impl WhisperBackend {
    pub fn new(model: Arc<WhisperModel>, state_pool_size: usize) -> Self {
        Self {
            pool: Arc::new(StatePool::new(Arc::clone(&model), state_pool_size.max(1))),
            model,
        }
    }

    pub fn load(config: WhisperConfig) -> Result<Self, AsrError> {
        let pool_size = config.state_pool_size;
        Ok(Self::new(Arc::new(WhisperModel::load(config)?), pool_size))
    }

    pub fn model(&self) -> &Arc<WhisperModel> {
        &self.model
    }
}

impl AsrBackend for WhisperBackend {
    fn transcribe(
        &self,
        pcm: &[f32],
        mode: DecodeMode,
        prompt: Option<&str>,
        language: Option<&str>,
    ) -> Result<TranscriptResult, AsrError> {
        let mut state = self.pool.acquire()?;
        transcribe(&self.model, state.get_mut(), pcm, mode, prompt, language)
    }

    fn n_threads(&self) -> usize {
        self.model.config().n_threads.max(1) as usize
    }

    fn name(&self) -> &'static str {
        "whisper"
    }
}

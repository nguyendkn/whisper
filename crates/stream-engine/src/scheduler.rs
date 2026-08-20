use std::sync::Arc;
use std::time::Instant;

use whisper_core::{AsrBackend, DecodeMode, TranscriptResult, WhisperBackend, WhisperModel};

use crate::{budget::ThreadBudget, error::EngineError};

/// Xếp hàng inference cho **một** backend ASR (whisper.cpp, zipformer...), dưới
/// hạn mức thread dùng chung ([`ThreadBudget`]).
///
/// Nhiều scheduler (partial + final) phải dùng cùng một `ThreadBudget`, nếu không
/// chúng sẽ cộng dồn thread và làm sập hiệu năng.
pub struct InferenceScheduler {
    backend: Arc<dyn AsrBackend>,
    budget: Arc<ThreadBudget>,
    threads: usize,
}

impl InferenceScheduler {
    /// Scheduler đứng một mình cho whisper: tự tạo hạn mức
    /// `max_concurrent * n_threads`. Giữ nguyên chữ ký cũ cho cli/test.
    pub fn new(model: Arc<WhisperModel>, max_concurrent: usize) -> Self {
        let max_concurrent = max_concurrent.max(1);
        let threads = model.config().n_threads.max(1) as usize;
        let budget = ThreadBudget::new(max_concurrent * threads);
        Self::with_budget(Arc::new(WhisperBackend::new(model, max_concurrent)), budget)
    }

    /// Scheduler dùng chung hạn mức với các backend khác trong tiến trình.
    pub fn with_budget(backend: Arc<dyn AsrBackend>, budget: Arc<ThreadBudget>) -> Self {
        let threads = backend.n_threads().max(1);
        Self {
            backend,
            budget,
            threads,
        }
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Số lượt inference của model này chạy song song được trong hạn mức.
    pub fn max_concurrent(&self) -> usize {
        (self.budget.total() / self.threads).max(1)
    }

    /// Số thread còn rảnh trong hạn mức — dùng để log độ sâu hàng đợi.
    pub fn available_permits(&self) -> usize {
        self.budget.available()
    }

    pub fn budget(&self) -> &Arc<ThreadBudget> {
        &self.budget
    }

    /// Xếp hàng một cửa sổ PCM 16 kHz mono và chờ kết quả.
    ///
    /// whisper.cpp là tải CPU/GPU đồng bộ nên phần chạy nằm trong
    /// `spawn_blocking`; permit thread được giữ suốt lượt chạy.
    pub async fn submit(
        &self,
        pcm: Vec<f32>,
        mode: DecodeMode,
        prompt: Option<String>,
        language: Option<String>,
    ) -> Result<TranscriptResult, EngineError> {
        let queued_at = Instant::now();
        let _permit = self.budget.acquire(self.threads).await?;
        let queue_ms = queued_at.elapsed().as_millis() as u64;
        if queue_ms > 50 {
            tracing::debug!(
                queue_ms,
                ?mode,
                threads = self.threads,
                "inference chờ hạn mức thread"
            );
        }

        let backend = Arc::clone(&self.backend);
        let result = tokio::task::spawn_blocking(move || {
            backend.transcribe(&pcm, mode, prompt.as_deref(), language.as_deref())
        })
        .await
        .map_err(|e| EngineError::Join(e.to_string()))?;

        Ok(result?)
    }
}

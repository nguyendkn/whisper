use std::sync::Arc;
use std::time::Instant;

use whisper_core::{transcribe, DecodeMode, StatePool, TranscriptResult, WhisperModel};

use crate::{budget::ThreadBudget, error::EngineError};

/// Xếp hàng inference cho **một** model, dưới hạn mức thread dùng chung
/// ([`ThreadBudget`]) và với pool `WhisperState` để không cấp phát lại KV cache.
///
/// Nhiều scheduler (partial + final) phải dùng cùng một `ThreadBudget`, nếu không
/// chúng sẽ cộng dồn thread và làm sập hiệu năng.
#[derive(Debug)]
pub struct InferenceScheduler {
    pool: Arc<StatePool>,
    budget: Arc<ThreadBudget>,
    threads: usize,
}

impl InferenceScheduler {
    /// Scheduler đứng một mình: tự tạo hạn mức `max_concurrent * n_threads`.
    pub fn new(model: Arc<WhisperModel>, max_concurrent: usize) -> Self {
        let max_concurrent = max_concurrent.max(1);
        let threads = model.config().n_threads.max(1) as usize;
        let budget = ThreadBudget::new(max_concurrent * threads);
        Self::with_budget(model, budget, max_concurrent)
    }

    /// Scheduler dùng chung hạn mức với các model khác trong tiến trình.
    pub fn with_budget(
        model: Arc<WhisperModel>,
        budget: Arc<ThreadBudget>,
        state_pool_size: usize,
    ) -> Self {
        let threads = model.config().n_threads.max(1) as usize;
        Self {
            pool: Arc::new(StatePool::new(model, state_pool_size.max(1))),
            budget,
            threads,
        }
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

    pub fn model(&self) -> &Arc<WhisperModel> {
        self.pool.model()
    }

    /// Xếp hàng một cửa sổ PCM 16 kHz mono và chờ kết quả.
    ///
    /// whisper.cpp là tải CPU/GPU đồng bộ nên phần chạy nằm trong
    /// `spawn_blocking`; permit thread được giữ suốt lượt chạy.
    pub async fn submit(
        &self,
        pcm: Vec<f32>,
        mode: DecodeMode,
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

        let pool = Arc::clone(&self.pool);
        let result = tokio::task::spawn_blocking(move || {
            let mut state = pool.acquire()?;
            let model = Arc::clone(pool.model());
            transcribe(&model, state.get_mut(), &pcm, mode)
        })
        .await
        .map_err(|e| EngineError::Join(e.to_string()))?;

        Ok(result?)
    }
}

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Semaphore;
use whisper_core::{transcribe, DecodeMode, StatePool, TranscriptResult, WhisperModel};

use crate::error::EngineError;

/// Giới hạn số inference chạy đồng thời trên toàn hệ thống (không phải
/// per-session) và giữ pool `WhisperState` để không cấp phát lại KV cache.
///
/// `max_concurrent` × `n_threads` của model nên ≤ số core vật lý: vượt quá thì
/// các lượt inference tranh CPU và RTF của **mọi** session cùng xấu đi. Trên NUC
/// thường là 1–2 permit; trên H100 giới hạn thực tế là VRAM.
#[derive(Debug)]
pub struct InferenceScheduler {
    pool: Arc<StatePool>,
    semaphore: Arc<Semaphore>,
    max_concurrent: usize,
}

impl InferenceScheduler {
    pub fn new(model: Arc<WhisperModel>, max_concurrent: usize) -> Self {
        let max_concurrent = max_concurrent.max(1);
        Self {
            pool: Arc::new(StatePool::new(model, max_concurrent)),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }

    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Số permit còn rảnh — dùng để log queue depth.
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    pub fn model(&self) -> &Arc<WhisperModel> {
        self.pool.model()
    }

    /// Xếp hàng một cửa sổ PCM 16 kHz mono và chờ kết quả.
    ///
    /// whisper.cpp là tải CPU/GPU đồng bộ nên phần chạy nằm trong
    /// `spawn_blocking`; permit của semaphore được giữ suốt lượt chạy.
    pub async fn submit(
        &self,
        pcm: Vec<f32>,
        mode: DecodeMode,
    ) -> Result<TranscriptResult, EngineError> {
        let queued_at = Instant::now();
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| EngineError::Shutdown)?;
        let queue_ms = queued_at.elapsed().as_millis() as u64;
        if queue_ms > 50 {
            tracing::debug!(queue_ms, ?mode, "inference waited for a permit");
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

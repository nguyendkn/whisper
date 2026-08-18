use std::sync::Arc;

use tokio::sync::{Semaphore, SemaphorePermit};

use crate::error::EngineError;

/// Hạn mức thread CPU dùng chung cho **mọi** model trong tiến trình.
///
/// Vì sao cần: mỗi scheduler có semaphore riêng thì hai model (một cho partial,
/// một cho final) sẽ chạy đồng thời và cộng dồn số thread. Đo thực tế trên máy 16
/// core: turbo 12 thread + base 12 thread chạy song song làm cả file 128 s tụt lại
/// 93 s so với realtime — tệ hơn cả khi chỉ dùng turbo. ggml spin-wait ở barrier nên
/// oversubscribe không chậm dần mà sập.
///
/// Mỗi lượt inference xin đúng `n_threads` permit, nên bất biến
/// `tổng thread đang chạy <= total` được giữ ở cấp tiến trình.
#[derive(Debug)]
pub struct ThreadBudget {
    semaphore: Semaphore,
    total: usize,
}

impl ThreadBudget {
    pub fn new(total: usize) -> Arc<Self> {
        let total = total.max(1);
        Arc::new(Self {
            semaphore: Semaphore::new(total),
            total,
        })
    }

    /// Chừa 2 core cho tokio, VAD, decode audio và hệ điều hành — đặt bằng đúng số
    /// core là cấu hình chậm nhất đo được.
    pub fn auto() -> Arc<Self> {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self::new(cores.saturating_sub(2).max(1))
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn available(&self) -> usize {
        self.semaphore.available_permits()
    }

    /// Giữ `threads` permit tới khi guard bị drop.
    pub(crate) async fn acquire(&self, threads: usize) -> Result<SemaphorePermit<'_>, EngineError> {
        let want = threads.clamp(1, self.total) as u32;
        self.semaphore
            .acquire_many(want)
            .await
            .map_err(|_| EngineError::Shutdown)
    }
}

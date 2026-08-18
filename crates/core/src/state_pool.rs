use std::sync::{Arc, Mutex};

use whisper_rs::WhisperState;

use crate::{error::AsrError, model::WhisperModel};

/// Pool các `WhisperState` để không phải cấp phát KV cache mỗi lượt inference.
///
/// Kích thước pool nên bằng số inference song song tối đa (số permit của
/// scheduler): mỗi permit giữ tối đa một state, nên pool không phình thêm.
#[derive(Debug)]
pub struct StatePool {
    model: Arc<WhisperModel>,
    idle: Mutex<Vec<WhisperState>>,
    capacity: usize,
}

impl StatePool {
    pub fn new(model: Arc<WhisperModel>, capacity: usize) -> Self {
        Self {
            model,
            idle: Mutex::new(Vec::with_capacity(capacity)),
            capacity: capacity.max(1),
        }
    }

    /// Mượn một state. Trả về pool khi guard bị drop (nếu pool chưa đầy).
    pub fn acquire(self: &Arc<Self>) -> Result<PooledState, AsrError> {
        let reused = self.idle.lock().expect("state pool poisoned").pop();
        let state = match reused {
            Some(state) => state,
            None => {
                tracing::debug!("allocating new whisper state");
                self.model.create_state()?
            }
        };
        Ok(PooledState {
            state: Some(state),
            pool: Arc::clone(self),
        })
    }

    pub fn model(&self) -> &Arc<WhisperModel> {
        &self.model
    }

    fn release(&self, state: WhisperState) {
        let mut idle = self.idle.lock().expect("state pool poisoned");
        if idle.len() < self.capacity {
            idle.push(state);
        }
    }
}

/// Guard trả state về pool khi drop.
pub struct PooledState {
    state: Option<WhisperState>,
    pool: Arc<StatePool>,
}

impl PooledState {
    pub fn get_mut(&mut self) -> &mut WhisperState {
        self.state.as_mut().expect("state taken twice")
    }
}

impl Drop for PooledState {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            self.pool.release(state);
        }
    }
}

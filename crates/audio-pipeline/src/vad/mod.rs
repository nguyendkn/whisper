//! VAD tách làm hai phần: máy trạng thái thuần ([`SpeechGate`]) và bộ tính xác
//! suất có tiếng nói ([`SpeechProbe`]). Nhờ vậy test được logic cắt câu mà
//! không cần load model.

mod energy;
mod gate;
#[cfg(feature = "vad-silero")]
mod silero;

pub use energy::EnergyVad;
pub use gate::{GateConfig, SpeechGate, VadEvent};
#[cfg(feature = "vad-silero")]
pub use silero::SileroVad;

use crate::error::AudioError;

/// Nguồn xác suất có tiếng nói cho một đoạn PCM 16 kHz mono.
pub trait SpeechProbe: Send {
    /// Số sample mỗi frame mà bộ này chấm điểm.
    fn frame_samples(&self) -> usize;

    /// Xác suất từng frame trong `pcm`.
    fn probabilities(&mut self, pcm: &[f32]) -> Result<Vec<f32>, AudioError>;

    /// Một điểm duy nhất cho cả chunk. Lấy max: chỉ cần một frame có tiếng là
    /// chunk đó không phải khoảng lặng.
    fn speech_probability(&mut self, pcm: &[f32]) -> Result<f32, AudioError> {
        let probs = self.probabilities(pcm)?;
        Ok(probs.into_iter().fold(0.0f32, f32::max))
    }

    /// Độ dài một frame theo ms — dùng để quy đổi ngưỡng im lặng sang số frame.
    fn frame_ms(&self) -> u32 {
        (self.frame_samples() as u64 * 1_000 / super::TARGET_SAMPLE_RATE as u64) as u32
    }
}

//! Đường ống audio: capture, resample về 16 kHz mono, VAD và sliding-window
//! buffer. Không biết gì về ASR — dùng được cho cả batch lẫn streaming.

#[cfg(feature = "capture")]
pub mod capture;
pub mod error;
pub mod resampler;
pub mod ring_buffer;
pub mod vad;

#[cfg(feature = "capture")]
pub use capture::MicCapture;
pub use error::AudioError;
pub use resampler::{pcm_i16_le_to_f32, AudioResampler, TARGET_SAMPLE_RATE};
pub use ring_buffer::AudioRingBuffer;
#[cfg(feature = "vad-silero")]
pub use vad::SileroVad;
pub use vad::{EnergyVad, GateConfig, SpeechGate, SpeechProbe, VadEvent};

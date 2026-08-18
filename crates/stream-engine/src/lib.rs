//! Lớp orchestration: buffer → VAD → ASR → partial/final. Đây là nơi giới hạn
//! tài nguyên (số inference song song), không phải ở server hay ở core.

pub mod config;
pub mod error;
pub mod event;
pub mod scheduler;
pub mod session;
pub mod transcript;

pub use config::SessionConfig;
pub use error::EngineError;
pub use event::{StreamEvent, TranscriptUpdate};
pub use scheduler::InferenceScheduler;
pub use session::Session;
pub use transcript::{CommitOutcome, Transcript};

use std::sync::Arc;

use audio_pipeline::{EnergyVad, SpeechProbe};
use stream_engine::InferenceScheduler;
use whisper_core::WhisperModel;

use crate::config::ServerConfig;

#[derive(Clone)]
pub struct AppState {
    pub scheduler: Arc<InferenceScheduler>,
    pub cfg: Arc<ServerConfig>,
}

impl AppState {
    pub fn init(cfg: ServerConfig) -> anyhow::Result<Self> {
        let model = Arc::new(WhisperModel::load(cfg.whisper_config())?);
        let scheduler = Arc::new(InferenceScheduler::new(model, cfg.max_concurrent_inference));
        Ok(Self {
            scheduler,
            cfg: Arc::new(cfg),
        })
    }

    /// Mỗi session cần một VAD riêng vì `detect_speech` giữ state nội bộ.
    /// Model Silero chỉ vài MB nên load per-connection là chấp nhận được; nếu
    /// số kết nối/giây cao thì đổi sang pool giống `StatePool`.
    pub fn new_probe(&self) -> Box<dyn SpeechProbe> {
        match self.silero_probe() {
            Some(probe) => probe,
            None => Box::new(EnergyVad::default()),
        }
    }

    #[cfg(feature = "vad-silero")]
    fn silero_probe(&self) -> Option<Box<dyn SpeechProbe>> {
        let path = self.cfg.vad_model_path()?;
        match audio_pipeline::SileroVad::load(&path, self.cfg.vad.n_threads, self.cfg.vad.use_gpu) {
            Ok(vad) => Some(Box::new(vad)),
            Err(err) => {
                tracing::error!(%err, "silero VAD unavailable, falling back to energy VAD");
                None
            }
        }
    }

    #[cfg(not(feature = "vad-silero"))]
    fn silero_probe(&self) -> Option<Box<dyn SpeechProbe>> {
        None
    }
}

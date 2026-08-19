use std::collections::HashMap;
use std::sync::Arc;

use audio_pipeline::{EnergyVad, GatedProbe, SpeechProbe};
use stream_engine::{InferenceScheduler, SessionEngines, ThreadBudget};
use whisper_core::WhisperModel;

use crate::config::ServerConfig;

#[derive(Clone)]
pub struct AppState {
    pub scheduler: Arc<InferenceScheduler>,
    /// Scheduler riêng cho partial nếu config khai báo `[partial_model]`.
    pub partial_scheduler: Option<Arc<InferenceScheduler>>,
    /// Model riêng theo ngôn ngữ cho lượt Final; ngôn ngữ không có trong map thì
    /// dùng model chính.
    pub language_schedulers: Arc<HashMap<String, Arc<InferenceScheduler>>>,
    pub cfg: Arc<ServerConfig>,
}

impl AppState {
    pub fn init(cfg: ServerConfig) -> anyhow::Result<Self> {
        // Một hạn mức thread duy nhất cho mọi model: model partial và model final
        // không được cộng dồn thread, nếu không cả hai cùng chậm.
        let budget = if cfg.cpu_thread_budget > 0 {
            ThreadBudget::new(cfg.cpu_thread_budget)
        } else {
            ThreadBudget::auto()
        };

        let model = Arc::new(WhisperModel::load(cfg.whisper_config())?);
        let scheduler = Arc::new(InferenceScheduler::with_budget(
            model,
            Arc::clone(&budget),
            cfg.max_concurrent_inference,
        ));
        // Model partial (tuỳ chọn): model nhỏ giữ độ trễ partial thấp trong khi
        // model chính vẫn là model lớn cho lượt chốt câu.
        let partial_scheduler = match cfg.partial_whisper_config() {
            Some(partial_cfg) => {
                let partial_model = Arc::new(WhisperModel::load(partial_cfg)?);
                Some(Arc::new(InferenceScheduler::with_budget(
                    partial_model,
                    Arc::clone(&budget),
                    cfg.max_concurrent_partial_inference,
                )))
            }
            None => None,
        };

        // Model theo ngôn ngữ: không model nào thắng ở mọi thứ tiếng, nên cho phép
        // khai báo riêng và định tuyến theo `?language=` của session.
        let mut language_schedulers: HashMap<String, Arc<InferenceScheduler>> = HashMap::new();
        for (languages, whisper_cfg) in cfg.language_whisper_configs() {
            let model = Arc::new(WhisperModel::load(whisper_cfg)?);
            let scheduler = Arc::new(InferenceScheduler::with_budget(
                model,
                Arc::clone(&budget),
                cfg.max_concurrent_inference,
            ));
            for language in languages {
                language_schedulers.insert(language.to_lowercase(), Arc::clone(&scheduler));
            }
        }

        tracing::info!(
            cpu_thread_budget = budget.total(),
            language_models = language_schedulers.len(),
            partial_model = partial_scheduler.is_some(),
            "inference budget ready"
        );

        Ok(Self {
            scheduler,
            partial_scheduler,
            language_schedulers: Arc::new(language_schedulers),
            cfg: Arc::new(cfg),
        })
    }

    /// Model cho một session. `language` lấy từ query của client; không khai báo
    /// riêng thì dùng model chính.
    pub fn engines(&self, language: Option<&str>) -> SessionEngines {
        let finals = language
            .map(str::to_lowercase)
            .and_then(|language| self.language_schedulers.get(&language).cloned())
            .unwrap_or_else(|| Arc::clone(&self.scheduler));
        SessionEngines {
            finals,
            partials: self.partial_scheduler.clone(),
        }
    }

    /// Mỗi session cần một VAD riêng vì `detect_speech` giữ state nội bộ.
    /// Model Silero chỉ vài MB nên load per-connection là chấp nhận được; nếu
    /// số kết nối/giây cao thì đổi sang pool giống `StatePool`.
    pub fn new_probe(&self) -> Box<dyn SpeechProbe> {
        match self.silero_probe() {
            // Cổng năng lượng đứng trước Silero: khoảng lặng không phải trả tiền
            // cho một lượt inference VAD.
            Some(probe) if self.cfg.vad.energy_gate_threshold > 0.0 => {
                Box::new(GatedProbe::new(probe, self.cfg.vad.energy_gate_threshold))
            }
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

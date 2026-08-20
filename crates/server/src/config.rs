use std::path::PathBuf;

use audio_pipeline::GateConfig;
use serde::Deserialize;
use stream_engine::SessionConfig;
use whisper_core::WhisperConfig;

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub bind_addr: String,
    /// Kích thước pool `WhisperState` của model chính. Số lượt chạy song song thực
    /// tế do `cpu_thread_budget` quyết định, không phải trường này.
    pub max_concurrent_inference: usize,
    /// Tổng số thread CPU cho toàn bộ inference trong tiến trình (mọi model dùng
    /// chung). 0 = tự lấy `số core - 2`.
    #[serde(default)]
    pub cpu_thread_budget: usize,
    pub model: ModelSettings,
    /// Model nhỏ chạy partial. Bỏ trống = dùng luôn model chính.
    #[serde(default)]
    pub partial_model: Option<ModelSettings>,
    /// Model riêng theo ngôn ngữ, cho lượt `Final`.
    ///
    /// Cần thiết vì không có model nào thắng ở mọi thứ tiếng: đo trên FLEURS,
    /// `large-v3` tốt hơn cho tiếng Anh (WER 3,90% so với 4,75%) nhưng **tệ hơn**
    /// cho tiếng Việt (11,43% so với 9,39% của `large-v3-turbo`).
    #[serde(default)]
    pub language_models: Vec<LanguageModel>,
    /// Hạn mức song song riêng cho model partial.
    #[serde(default = "default_partial_concurrency")]
    pub max_concurrent_partial_inference: usize,
    pub vad: VadSettings,
    pub session: SessionSettings,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSettings {
    /// "whisper" (mặc định) hoặc "zipformer" (RNN-T qua sherpa-onnx; `path` là
    /// thư mục chứa encoder/decoder/joiner .onnx + tokens.txt).
    #[serde(default)]
    pub engine: String,
    /// Zipformer: dùng bản .int8.onnx.
    #[serde(default)]
    pub quantized: bool,
    pub path: PathBuf,
    /// Rỗng = auto-detect ngôn ngữ.
    #[serde(default)]
    pub language: String,
    pub n_threads: i32,
    #[serde(default)]
    pub use_gpu: bool,
    #[serde(default)]
    pub gpu_device: i32,
    #[serde(default)]
    pub flash_attn: bool,
    /// 0 hoặc 1 = greedy. Beam chỉ áp dụng cho lượt `Final`.
    #[serde(default)]
    pub beam_size: i32,
    /// Bước tăng temperature khi whisper tự thấy kết quả tệ. 0 = tắt fallback.
    #[serde(default)]
    pub temperature_inc: f32,
    pub min_audio_ms: u32,
    /// Bỏ segment có xác suất "không có tiếng nói" cao hơn ngưỡng (chặn ảo giác).
    #[serde(default = "default_no_speech_thold")]
    pub no_speech_thold: f32,
    /// Bỏ segment có độ tự tin trung bình dưới ngưỡng (0.0 = tắt).
    #[serde(default)]
    pub min_confidence: f32,
    #[serde(default)]
    pub initial_prompt: String,
    /// Thu nhỏ encoder context cho lượt partial theo đúng độ dài cửa sổ.
    #[serde(default = "default_true")]
    pub scale_partial_audio_ctx: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LanguageModel {
    /// Các mã ngôn ngữ dùng model này (ví dụ `["en"]`).
    pub languages: Vec<String>,
    #[serde(flatten)]
    pub model: ModelSettings,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VadSettings {
    /// Rỗng = dùng energy VAD (chỉ nên cho lúc dev).
    #[serde(default)]
    pub model_path: String,
    pub threshold: f32,
    pub silence_ms_for_end: u32,
    pub min_speech_ms: u32,
    pub n_threads: i32,
    #[serde(default)]
    pub use_gpu: bool,
    /// Cổng năng lượng đứng trước Silero: dưới ngưỡng này thì không chạy model VAD.
    /// 0 = tắt cổng. Phải thấp hơn `threshold` để không chặn mất tiếng nói thật.
    #[serde(default = "default_energy_gate")]
    pub energy_gate_threshold: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SessionSettings {
    pub max_utterance_secs: f32,
    pub partial_window_secs: f32,
    pub partial_interval_ms: u64,
    pub pre_roll_secs: f32,
    /// Mồi lượt Final bằng text đã chốt trước đó.
    #[serde(default)]
    pub condition_on_previous: bool,
    /// LocalAgreement-2 cho partial: chỉ hiện phần hai lượt decode liên tiếp đồng ý.
    #[serde(default = "default_true")]
    pub local_agreement: bool,
    #[serde(default = "default_probe_backlog")]
    pub max_probe_backlog_secs: f32,
}

impl ServerConfig {
    /// Đọc `config/default.toml` rồi override bằng biến môi trường
    /// `WHISPER_RT__MODEL__PATH=...` (hai dấu gạch dưới tách theo tầng).
    /// `WHISPER_RT_CONFIG` đổi được đường dẫn file để không phụ thuộc CWD.
    pub fn load() -> anyhow::Result<Self> {
        let path = std::env::var("WHISPER_RT_CONFIG").unwrap_or_else(|_| "config/default".into());
        let settings = config::Config::builder()
            .add_source(config::File::with_name(&path))
            .add_source(
                config::Environment::with_prefix("WHISPER_RT")
                    .separator("__")
                    .try_parsing(true),
            )
            .build()?;
        Ok(settings.try_deserialize()?)
    }

    pub fn whisper_config(&self) -> WhisperConfig {
        WhisperConfig {
            model_path: self.model.path.clone(),
            language: opt(&self.model.language),
            n_threads: self.model.n_threads,
            translate: false,
            beam_size: (self.model.beam_size > 1).then_some(self.model.beam_size),
            temperature: 0.0,
            temperature_inc: self.model.temperature_inc,
            entropy_thold: 2.4,
            logprob_thold: -1.0,
            use_gpu: self.model.use_gpu,
            gpu_device: self.model.gpu_device,
            flash_attn: self.model.flash_attn,
            initial_prompt: opt(&self.model.initial_prompt),
            min_audio_ms: self.model.min_audio_ms,
            no_speech_thold: self.model.no_speech_thold,
            min_confidence: self.model.min_confidence,
            state_pool_size: self.max_concurrent_inference.max(1),
            scale_partial_audio_ctx: self.model.scale_partial_audio_ctx,
            token_timestamps: self.session.local_agreement,
        }
    }

    pub(crate) fn model_settings_to_config(&self, model: &ModelSettings) -> WhisperConfig {
        WhisperConfig {
            model_path: model.path.clone(),
            language: opt(&model.language),
            n_threads: model.n_threads,
            translate: false,
            beam_size: (model.beam_size > 1).then_some(model.beam_size),
            temperature: 0.0,
            temperature_inc: model.temperature_inc,
            entropy_thold: 2.4,
            logprob_thold: -1.0,
            use_gpu: model.use_gpu,
            gpu_device: model.gpu_device,
            flash_attn: model.flash_attn,
            initial_prompt: opt(&model.initial_prompt),
            min_audio_ms: model.min_audio_ms,
            no_speech_thold: model.no_speech_thold,
            min_confidence: model.min_confidence,
            state_pool_size: self.max_concurrent_inference.max(1),
            scale_partial_audio_ctx: model.scale_partial_audio_ctx,
            token_timestamps: self.session.local_agreement,
        }
    }

    /// Config cho model partial, nếu có khai báo riêng.
    pub fn partial_whisper_config(&self) -> Option<WhisperConfig> {
        let model = self.partial_model.as_ref()?;
        Some(WhisperConfig {
            model_path: model.path.clone(),
            language: opt(&model.language).or_else(|| opt(&self.model.language)),
            n_threads: model.n_threads,
            translate: false,
            // Partial luôn greedy: beam search chỉ đáng cho lượt chốt câu.
            beam_size: None,
            temperature: 0.0,
            temperature_inc: 0.0,
            entropy_thold: 2.4,
            logprob_thold: -1.0,
            use_gpu: model.use_gpu,
            gpu_device: model.gpu_device,
            flash_attn: model.flash_attn,
            initial_prompt: opt(&model.initial_prompt),
            min_audio_ms: model.min_audio_ms,
            state_pool_size: self.max_concurrent_partial_inference.max(1),
            scale_partial_audio_ctx: model.scale_partial_audio_ctx,
            no_speech_thold: model.no_speech_thold,
            min_confidence: model.min_confidence,
            token_timestamps: self.session.local_agreement,
        })
    }

    pub fn session_config(&self) -> SessionConfig {
        SessionConfig {
            max_utterance_secs: self.session.max_utterance_secs,
            partial_window_secs: self.session.partial_window_secs,
            partial_interval_ms: self.session.partial_interval_ms,
            pre_roll_secs: self.session.pre_roll_secs,
            max_probe_backlog_secs: self.session.max_probe_backlog_secs,
            condition_on_previous: self.session.condition_on_previous,
            prompt_chars: 200,
            local_agreement: self.session.local_agreement,
            // Ngôn ngữ mặc định của session; WebSocket có `?language=` thì override.
            language: opt(&self.model.language),
            gate: GateConfig {
                threshold: self.vad.threshold,
                silence_ms_for_end: self.vad.silence_ms_for_end,
                min_speech_ms: self.vad.min_speech_ms,
            },
        }
    }

    pub fn vad_model_path(&self) -> Option<PathBuf> {
        opt(&self.vad.model_path).map(PathBuf::from)
    }
}

fn default_partial_concurrency() -> usize {
    1
}

fn default_probe_backlog() -> f32 {
    2.0
}

fn default_energy_gate() -> f32 {
    0.15
}

fn default_no_speech_thold() -> f32 {
    0.6
}

fn default_true() -> bool {
    true
}

fn opt(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

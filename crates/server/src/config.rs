use std::path::PathBuf;

use audio_pipeline::GateConfig;
use serde::Deserialize;
use stream_engine::SessionConfig;
use whisper_core::WhisperConfig;

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub bind_addr: String,
    /// Số inference song song trên toàn server. Trên NUC để 1–2; trên GPU lớn
    /// tăng dần rồi đo RTF thay vì đoán.
    pub max_concurrent_inference: usize,
    pub model: ModelSettings,
    pub vad: VadSettings,
    pub session: SessionSettings,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSettings {
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
    pub min_audio_ms: u32,
    #[serde(default)]
    pub initial_prompt: String,
    /// Thu nhỏ encoder context cho lượt partial theo đúng độ dài cửa sổ.
    #[serde(default = "default_true")]
    pub scale_partial_audio_ctx: bool,
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
}

#[derive(Debug, Clone, Deserialize)]
pub struct SessionSettings {
    pub max_utterance_secs: f32,
    pub partial_window_secs: f32,
    pub partial_interval_ms: u64,
    pub pre_roll_secs: f32,
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
            use_gpu: self.model.use_gpu,
            gpu_device: self.model.gpu_device,
            flash_attn: self.model.flash_attn,
            initial_prompt: opt(&self.model.initial_prompt),
            min_audio_ms: self.model.min_audio_ms,
            state_pool_size: self.max_concurrent_inference.max(1),
            scale_partial_audio_ctx: self.model.scale_partial_audio_ctx,
        }
    }

    pub fn session_config(&self) -> SessionConfig {
        SessionConfig {
            max_utterance_secs: self.session.max_utterance_secs,
            partial_window_secs: self.session.partial_window_secs,
            partial_interval_ms: self.session.partial_interval_ms,
            pre_roll_secs: self.session.pre_roll_secs,
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

fn default_true() -> bool {
    true
}

fn opt(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

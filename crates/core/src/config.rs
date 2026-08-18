use std::path::PathBuf;

/// Sample rate duy nhất whisper.cpp nhận.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Cửa sổ mà encoder của whisper luôn xử lý — mọi input đều được pad lên 30 s
/// trước khi vào encoder, nên chunk 2 s tốn gần bằng chunk 30 s.
pub const WHISPER_CHUNK_SECS: f32 = 30.0;

#[derive(Clone, Debug)]
pub struct WhisperConfig {
    pub model_path: PathBuf,
    /// `None` = auto-detect (chậm hơn, thêm một lượt decode).
    pub language: Option<String>,
    pub n_threads: i32,
    pub translate: bool,
    /// `None` = greedy (dùng cho partial). Beam search chỉ nên bật cho `Final`.
    pub beam_size: Option<i32>,
    pub temperature: f32,
    pub use_gpu: bool,
    pub gpu_device: i32,
    pub flash_attn: bool,
    /// Prompt mồi để ổn định thuật ngữ riêng (tên sản phẩm, từ viết tắt).
    pub initial_prompt: Option<String>,
    /// Ngưỡng độ dài tối thiểu trước khi gọi inference.
    pub min_audio_ms: u32,
    /// Số `WhisperState` giữ sẵn trong pool. Nên bằng số inference song song.
    pub state_pool_size: usize,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/ggml-large-v3-turbo.bin"),
            language: Some("vi".into()),
            n_threads: 4,
            translate: false,
            beam_size: None,
            temperature: 0.0,
            use_gpu: true,
            gpu_device: 0,
            flash_attn: false,
            initial_prompt: None,
            min_audio_ms: 1_000,
            state_pool_size: 2,
        }
    }
}

impl WhisperConfig {
    pub fn min_samples(&self) -> usize {
        (WHISPER_SAMPLE_RATE as u64 * self.min_audio_ms as u64 / 1_000) as usize
    }
}

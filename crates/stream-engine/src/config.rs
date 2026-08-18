use audio_pipeline::GateConfig;

#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Chặn trên cho một lượt nói; quá dài thì chốt cưỡng bức để client không
    /// phải chờ vô hạn (và để cửa sổ không vượt 30 s của whisper).
    pub max_utterance_secs: f32,
    /// Partial chỉ decode đuôi này, không decode lại cả lượt nói — cửa sổ 20–30 s
    /// decode mỗi giây là cách nhanh nhất để RTF vượt 1.
    pub partial_window_secs: f32,
    pub partial_interval_ms: u64,
    /// Audio giữ lại trước khi VAD báo có tiếng, tránh cắt mất phụ âm đầu.
    pub pre_roll_secs: f32,
    pub gate: GateConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_utterance_secs: 25.0,
            partial_window_secs: 6.0,
            partial_interval_ms: 800,
            pre_roll_secs: 0.4,
            gate: GateConfig::default(),
        }
    }
}

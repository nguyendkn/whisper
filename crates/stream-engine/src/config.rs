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
    /// RealtimeSTT dùng 1 s cho việc này; 0,3–0,4 s đo được là hay cắt mất từ đầu
    /// câu vì bản thân VAD đã trễ vài chục ms và ta chấm điểm theo cụm 256 ms.
    pub pre_roll_secs: f32,
    /// Chặn trên cho lượng audio chờ VAD chấm điểm. Nếu VAD không theo kịp luồng
    /// vào (máy quá tải), bỏ phần cũ nhất thay vì để độ trễ phình vô hạn — tương
    /// đương `allowed_latency_limit` của RealtimeSTT.
    pub max_probe_backlog_secs: f32,
    pub gate: GateConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_utterance_secs: 25.0,
            partial_window_secs: 6.0,
            partial_interval_ms: 800,
            pre_roll_secs: 1.0,
            max_probe_backlog_secs: 2.0,
            gate: GateConfig::default(),
        }
    }
}

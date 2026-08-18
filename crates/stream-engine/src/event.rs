use uuid::Uuid;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Partial(TranscriptUpdate),
    Final(TranscriptUpdate),
    Error { session_id: Uuid, message: String },
}

#[derive(Debug, Clone)]
pub struct TranscriptUpdate {
    pub session_id: Uuid,
    /// Lượt nói thứ mấy trong session — partial về muộn hơn final của cùng
    /// utterance thì client bỏ được nhờ số này.
    pub utterance: u64,
    /// Text của riêng lần decode này.
    pub text: String,
    /// Toàn văn: các lượt đã chốt + đuôi hiện tại.
    pub full_text: String,
    pub audio_ms: u32,
    pub rtf: f32,
}

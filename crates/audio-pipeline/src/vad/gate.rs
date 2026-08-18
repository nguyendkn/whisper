#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// Bắt đầu một lượt nói mới.
    SpeechStart,
    /// Đang nói (kể cả khoảng lặng ngắn chưa đủ để đóng câu).
    SpeechContinue,
    /// Im lặng đủ lâu — đóng câu, chạy `Final`.
    SpeechEnd,
    /// Chưa nói gì; không chạy inference để khỏi đốt compute.
    Silence,
}

#[derive(Debug, Clone, Copy)]
pub struct GateConfig {
    /// Ngưỡng xác suất coi là có tiếng nói.
    pub threshold: f32,
    /// Im lặng bao lâu thì đóng câu.
    pub silence_ms_for_end: u32,
    /// Phải có tiếng liên tục tối thiểu bao lâu mới coi là bắt đầu nói — chặn
    /// tiếng click, tiếng gõ bàn phím mở một lượt nói rỗng.
    pub min_speech_ms: u32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            silence_ms_for_end: 800,
            min_speech_ms: 128,
        }
    }
}

/// Máy trạng thái quyết định khi nào cắt cửa sổ gửi cho ASR. Thuần logic,
/// không phụ thuộc model nào.
#[derive(Debug)]
pub struct SpeechGate {
    config: GateConfig,
    silence_frames_for_end: u32,
    min_speech_frames: u32,
    consecutive_silence: u32,
    consecutive_speech: u32,
    is_speaking: bool,
}

impl SpeechGate {
    /// `frame_ms` lấy từ [`crate::vad::SpeechProbe::frame_ms`] — Silero chấm
    /// theo frame 512 sample (32 ms), đừng đoán bằng 30 ms.
    pub fn new(config: GateConfig, frame_ms: u32) -> Self {
        let frame_ms = frame_ms.max(1);
        Self {
            silence_frames_for_end: (config.silence_ms_for_end / frame_ms).max(1),
            min_speech_frames: (config.min_speech_ms / frame_ms).max(1),
            config,
            consecutive_silence: 0,
            consecutive_speech: 0,
            is_speaking: false,
        }
    }

    pub fn is_speaking(&self) -> bool {
        self.is_speaking
    }

    /// Nạp xác suất của một frame, nhận lại sự kiện tương ứng.
    pub fn observe(&mut self, speech_prob: f32) -> VadEvent {
        if speech_prob >= self.config.threshold {
            self.consecutive_silence = 0;
            self.consecutive_speech += 1;
            if !self.is_speaking {
                if self.consecutive_speech < self.min_speech_frames {
                    return VadEvent::Silence;
                }
                self.is_speaking = true;
                return VadEvent::SpeechStart;
            }
            return VadEvent::SpeechContinue;
        }

        self.consecutive_speech = 0;
        if !self.is_speaking {
            return VadEvent::Silence;
        }
        self.consecutive_silence += 1;
        if self.consecutive_silence >= self.silence_frames_for_end {
            self.is_speaking = false;
            self.consecutive_silence = 0;
            return VadEvent::SpeechEnd;
        }
        // Khoảng lặng giữa câu — vẫn coi là đang nói.
        VadEvent::SpeechContinue
    }

    /// Đóng câu chủ động (client ngắt kết nối, buffer đầy).
    pub fn force_end(&mut self) -> Option<VadEvent> {
        if !self.is_speaking {
            return None;
        }
        self.is_speaking = false;
        self.consecutive_silence = 0;
        self.consecutive_speech = 0;
        Some(VadEvent::SpeechEnd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> SpeechGate {
        SpeechGate::new(
            GateConfig {
                threshold: 0.5,
                silence_ms_for_end: 96,
                min_speech_ms: 64,
            },
            32,
        )
    }

    #[test]
    fn short_blip_does_not_open_an_utterance() {
        let mut gate = gate();
        assert_eq!(gate.observe(0.9), VadEvent::Silence);
        assert_eq!(gate.observe(0.1), VadEvent::Silence);
        assert!(!gate.is_speaking());
    }

    #[test]
    fn speech_then_silence_closes_the_utterance() {
        let mut gate = gate();
        assert_eq!(gate.observe(0.9), VadEvent::Silence);
        assert_eq!(gate.observe(0.9), VadEvent::SpeechStart);
        assert_eq!(gate.observe(0.9), VadEvent::SpeechContinue);
        // Hai frame lặng đầu vẫn trong grace period.
        assert_eq!(gate.observe(0.0), VadEvent::SpeechContinue);
        assert_eq!(gate.observe(0.0), VadEvent::SpeechContinue);
        assert_eq!(gate.observe(0.0), VadEvent::SpeechEnd);
        assert_eq!(gate.observe(0.0), VadEvent::Silence);
    }

    #[test]
    fn force_end_only_fires_while_speaking() {
        let mut gate = gate();
        assert_eq!(gate.force_end(), None);
        gate.observe(0.9);
        gate.observe(0.9);
        assert_eq!(gate.force_end(), Some(VadEvent::SpeechEnd));
    }
}

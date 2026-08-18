use crate::{error::AudioError, vad::EnergyVad, vad::SpeechProbe};

/// VAD hai tầng: cổng năng lượng rẻ đứng trước, model nặng chỉ chạy khi cổng mở.
///
/// Ý tưởng lấy từ RealtimeSTT (WebRTC VAD làm cổng, Silero xác nhận). Ở đây tầng
/// một là [`EnergyVad`] — gần như miễn phí và tự học noise floor — nên trên khoảng
/// lặng ta bỏ hẳn được lượt chạy Silero (~7–8 ms cho mỗi 256 ms audio mỗi session).
/// Tầng hai vẫn là thứ quyết định, nên độ chính xác không đổi khi có tiếng nói.
pub struct GatedProbe<P: SpeechProbe> {
    gate: EnergyVad,
    /// Dưới ngưỡng này coi như im lặng và không gọi tầng hai. Đặt thấp hơn ngưỡng
    /// của `SpeechGate` để cổng không bao giờ chặn mất tiếng nói thật.
    gate_threshold: f32,
    inner: P,
}

impl<P: SpeechProbe> GatedProbe<P> {
    pub fn new(inner: P, gate_threshold: f32) -> Self {
        Self {
            gate: EnergyVad::new(inner.frame_samples(), 4.0),
            gate_threshold,
            inner,
        }
    }
}

impl<P: SpeechProbe> SpeechProbe for GatedProbe<P> {
    fn frame_samples(&self) -> usize {
        self.inner.frame_samples()
    }

    fn probabilities(&mut self, pcm: &[f32]) -> Result<Vec<f32>, AudioError> {
        let energy = self.gate.probabilities(pcm)?;
        let loudest = energy.iter().copied().fold(0.0f32, f32::max);
        if loudest < self.gate_threshold {
            // Cổng đóng: trả về đúng số frame với xác suất 0, không chạy model.
            return Ok(vec![0.0; energy.len()]);
        }
        self.inner.probabilities(pcm)
    }
}

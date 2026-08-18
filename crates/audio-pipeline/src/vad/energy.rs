use crate::{error::AudioError, vad::SpeechProbe};

const DEFAULT_FRAME_SAMPLES: usize = 512;
/// Noise floor tụt nhanh khi gặp frame êm hơn, dâng chậm khi gặp frame to hơn —
/// nếu dâng nhanh thì chính giọng nói sẽ bị học vào nền.
const FLOOR_ATTACK: f32 = 0.05;
const FLOOR_DECAY: f32 = 0.001;

/// VAD theo năng lượng, noise floor thích ứng.
///
/// Không cần model, chi phí gần bằng không — nhưng chỉ phân biệt được to/nhỏ so
/// với nền: nhạc, quạt, tiếng gõ bàn phím đều bị tính là tiếng nói. Dùng làm
/// fallback khi chưa có model VAD; production nên dùng [`super::SileroVad`].
#[derive(Debug)]
pub struct EnergyVad {
    frame_samples: usize,
    /// Tỉ lệ RMS/noise-floor để coi là chắc chắn có tiếng nói.
    snr_target: f32,
    /// `None` cho tới frame đầu tiên: nền được khởi tạo bằng chính frame đó.
    noise_floor: Option<f32>,
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::new(DEFAULT_FRAME_SAMPLES, 4.0)
    }
}

impl EnergyVad {
    pub fn new(frame_samples: usize, snr_target: f32) -> Self {
        Self {
            frame_samples: frame_samples.max(1),
            snr_target: snr_target.max(1.001),
            noise_floor: None,
        }
    }

    fn score(&mut self, frame: &[f32]) -> f32 {
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        let Some(floor) = self.noise_floor else {
            // Chưa biết nền: lấy frame đầu làm nền, chưa kết luận gì.
            self.noise_floor = Some(rms.max(1e-6));
            return 0.0;
        };

        let alpha = if rms < floor {
            FLOOR_ATTACK
        } else {
            FLOOR_DECAY
        };
        let floor = ((1.0 - alpha) * floor + alpha * rms).max(1e-6);
        self.noise_floor = Some(floor);

        (rms / floor / self.snr_target).clamp(0.0, 1.0)
    }
}

impl SpeechProbe for EnergyVad {
    fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    fn probabilities(&mut self, pcm: &[f32]) -> Result<Vec<f32>, AudioError> {
        if pcm.len() < self.frame_samples {
            return Err(AudioError::VadFrameSize {
                got: pcm.len(),
                expected: self.frame_samples,
            });
        }
        Ok(pcm
            .chunks_exact(self.frame_samples)
            .map(|frame| self.score(frame))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_noise_stays_below_the_speech_threshold() {
        let mut vad = EnergyVad::default();
        let noise = vec![0.0005; 512];
        vad.probabilities(&noise).unwrap();
        for _ in 0..20 {
            let prob = vad.probabilities(&noise).unwrap()[0];
            assert!(prob < 0.5, "steady noise scored {prob}");
        }
    }

    #[test]
    fn speech_over_the_same_noise_floor_scores_high() {
        let mut vad = EnergyVad::default();
        let noise = vec![0.0005; 512];
        vad.probabilities(&noise).unwrap();
        let quiet = vad.probabilities(&noise).unwrap()[0];
        let loud = vad.probabilities(&vec![0.3; 512]).unwrap()[0];
        assert!(loud > quiet, "loud={loud} quiet={quiet}");
        assert!(loud >= 0.9, "loud={loud}");
    }

    #[test]
    fn rejects_frames_shorter_than_the_frame_size() {
        let mut vad = EnergyVad::default();
        assert!(vad.probabilities(&[0.0; 16]).is_err());
    }
}

use rubato::{FftFixedIn, Resampler};

use crate::error::AudioError;

/// Sample rate mà whisper.cpp yêu cầu.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Số frame input mỗi lượt gọi rubato.
const CHUNK_FRAMES: usize = 1_024;

/// Downmix về mono rồi resample về 16 kHz.
///
/// rubato đòi đúng `CHUNK_FRAMES` frame mỗi lượt `process`, còn WebSocket và
/// cpal thì giao frame với độ dài tuỳ ý — nên [`Self::push`] tích luỹ vào
/// `pending` và chỉ gọi resampler khi đủ một chunk. Phần dư nằm lại chờ lượt sau
/// hoặc [`Self::flush`].
pub struct AudioResampler {
    inner: Option<FftFixedIn<f32>>,
    channels_in: usize,
    pending: Vec<f32>,
}

impl AudioResampler {
    pub fn new(input_rate: u32, channels_in: usize) -> Result<Self, AudioError> {
        if input_rate == 0 || channels_in == 0 {
            return Err(AudioError::Resample(format!(
                "invalid input format: {input_rate} Hz, {channels_in} channels"
            )));
        }

        // Downmix trước, resample sau: chỉ phải resample một kênh.
        let inner = if input_rate == TARGET_SAMPLE_RATE {
            None
        } else {
            Some(
                FftFixedIn::<f32>::new(
                    input_rate as usize,
                    TARGET_SAMPLE_RATE as usize,
                    CHUNK_FRAMES,
                    4,
                    1,
                )
                .map_err(|e| AudioError::Resample(e.to_string()))?,
            )
        };

        Ok(Self {
            inner,
            channels_in,
            pending: Vec::with_capacity(CHUNK_FRAMES * 2),
        })
    }

    pub fn is_passthrough(&self) -> bool {
        self.inner.is_none() && self.channels_in == 1
    }

    /// Nhận audio interleaved ở sample rate gốc, trả về phần mono 16 kHz đã
    /// sẵn sàng (có thể rỗng nếu chưa đủ một chunk).
    pub fn push(&mut self, interleaved: &[f32]) -> Result<Vec<f32>, AudioError> {
        let mono = self.downmix(interleaved);
        let Some(_) = self.inner.as_ref() else {
            return Ok(mono);
        };

        self.pending.extend_from_slice(&mono);
        let mut out = Vec::new();
        while self.pending.len() >= CHUNK_FRAMES {
            let chunk: Vec<f32> = self.pending.drain(..CHUNK_FRAMES).collect();
            let resampled = self
                .inner
                .as_mut()
                .expect("checked above")
                .process(&[chunk], None)
                .map_err(|e| AudioError::Resample(e.to_string()))?;
            out.extend_from_slice(&resampled[0]);
        }
        Ok(out)
    }

    /// Đẩy phần dư còn lại ra (gọi khi session kết thúc), rubato tự pad silence.
    pub fn flush(&mut self) -> Result<Vec<f32>, AudioError> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(Vec::new());
        };
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let chunk: Vec<f32> = std::mem::take(&mut self.pending);
        let resampled = inner
            .process_partial(Some(&[chunk]), None)
            .map_err(|e| AudioError::Resample(e.to_string()))?;
        Ok(resampled[0].clone())
    }

    fn downmix(&self, interleaved: &[f32]) -> Vec<f32> {
        if self.channels_in == 1 {
            return interleaved.to_vec();
        }
        interleaved
            .chunks_exact(self.channels_in)
            .map(|frame| frame.iter().sum::<f32>() / self.channels_in as f32)
            .collect()
    }
}

/// Giải mã PCM i16 little-endian (định dạng dây của WebSocket) sang f32 [-1, 1].
pub fn pcm_i16_le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / i16::MAX as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_keeps_samples_untouched() {
        let mut resampler = AudioResampler::new(TARGET_SAMPLE_RATE, 1).unwrap();
        assert!(resampler.is_passthrough());
        assert_eq!(resampler.push(&[0.5, -0.5]).unwrap(), vec![0.5, -0.5]);
    }

    #[test]
    fn stereo_is_downmixed_to_mono() {
        let mut resampler = AudioResampler::new(TARGET_SAMPLE_RATE, 2).unwrap();
        assert_eq!(
            resampler.push(&[1.0, 0.0, 0.4, 0.6]).unwrap(),
            vec![0.5, 0.5]
        );
    }

    #[test]
    fn resamples_48k_to_16k_with_arbitrary_frame_sizes() {
        let mut resampler = AudioResampler::new(48_000, 1).unwrap();
        let mut produced = 0;
        // 300 lượt push 160 frame = 48 000 frame = 1 s audio.
        for _ in 0..300 {
            produced += resampler.push(&vec![0.0; 160]).unwrap().len();
        }
        produced += resampler.flush().unwrap().len();
        // Cho phép lệch một chunk vì phần dư nằm trong buffer nội bộ.
        assert!(
            (15_000..=17_000).contains(&produced),
            "expected ~16000 samples, got {produced}"
        );
    }

    #[test]
    fn decodes_i16_little_endian() {
        let bytes = [0x00, 0x00, 0xff, 0x7f];
        let pcm = pcm_i16_le_to_f32(&bytes);
        assert_eq!(pcm[0], 0.0);
        assert!((pcm[1] - 1.0).abs() < 1e-6);
    }
}

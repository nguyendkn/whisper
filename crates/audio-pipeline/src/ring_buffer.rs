use std::collections::VecDeque;

/// Sliding-window buffer cho streaming, giữ PCM f32 16 kHz mono.
///
/// `stream-engine` push liên tục, rồi lấy [`Self::snapshot`] (cả câu, cho
/// `Final`) hoặc [`Self::tail`] (vài giây cuối, cho `Partial`).
#[derive(Debug)]
pub struct AudioRingBuffer {
    samples: VecDeque<f32>,
    sample_rate: u32,
    max_samples: usize,
}

impl AudioRingBuffer {
    pub fn new(sample_rate: u32, max_duration_secs: f32) -> Self {
        let max_samples = (sample_rate as f32 * max_duration_secs).round() as usize;
        Self {
            samples: VecDeque::with_capacity(max_samples),
            sample_rate,
            max_samples: max_samples.max(1),
        }
    }

    /// Đẩy thêm audio; phần cũ nhất bị bỏ khi vượt cửa sổ tối đa.
    pub fn push(&mut self, chunk: &[f32]) {
        self.samples.extend(chunk.iter().copied());
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
    }

    /// Toàn bộ cửa sổ hiện tại, liên tục trong bộ nhớ để đưa vào whisper.
    pub fn snapshot(&self) -> Vec<f32> {
        self.samples.iter().copied().collect()
    }

    /// `secs` giây gần nhất. Dùng cho partial: decode lại cả cửa sổ 20–30 s
    /// mỗi giây là cách nhanh nhất để RTF vượt 1.
    pub fn tail(&self, secs: f32) -> Vec<f32> {
        let want = (self.sample_rate as f32 * secs).round() as usize;
        let skip = self.samples.len().saturating_sub(want);
        self.samples.iter().skip(skip).copied().collect()
    }

    /// Giữ lại `secs` giây cuối, bỏ phần trước đó. Dùng khi mở một lượt nói
    /// mới: cần một chút pre-roll để không cắt mất phụ âm đầu.
    pub fn retain_tail(&mut self, secs: f32) {
        let keep = (self.sample_rate as f32 * secs).round() as usize;
        while self.samples.len() > keep {
            self.samples.pop_front();
        }
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    pub fn is_full(&self) -> bool {
        self.samples.len() >= self.max_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_oldest_samples_past_the_window() {
        let mut buffer = AudioRingBuffer::new(16_000, 1.0);
        buffer.push(&vec![0.1; 16_000]);
        buffer.push(&vec![0.2; 4_000]);

        assert_eq!(buffer.len(), 16_000);
        let snapshot = buffer.snapshot();
        assert_eq!(snapshot[0], 0.1);
        assert_eq!(snapshot[15_999], 0.2);
    }

    #[test]
    fn tail_returns_only_the_requested_duration() {
        let mut buffer = AudioRingBuffer::new(16_000, 4.0);
        buffer.push(&vec![0.0; 48_000]);

        assert_eq!(buffer.tail(1.0).len(), 16_000);
        assert_eq!(buffer.tail(10.0).len(), buffer.len());
    }
}

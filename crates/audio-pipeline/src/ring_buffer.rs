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
    /// Số sample đã bị bỏ khỏi đầu buffer — cần để quy đổi mốc thời gian tuyệt đối
    /// (tính từ đầu lượt nói) sang vị trí trong buffer.
    dropped: u64,
}

impl AudioRingBuffer {
    pub fn new(sample_rate: u32, max_duration_secs: f32) -> Self {
        let max_samples = (sample_rate as f32 * max_duration_secs).round() as usize;
        Self {
            samples: VecDeque::with_capacity(max_samples),
            sample_rate,
            max_samples: max_samples.max(1),
            dropped: 0,
        }
    }

    /// Đẩy thêm audio; phần cũ nhất bị bỏ khi vượt cửa sổ tối đa.
    pub fn push(&mut self, chunk: &[f32]) {
        self.samples.extend(chunk.iter().copied());
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
            self.dropped += 1;
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
            // Phải đếm như `push` tràn cửa sổ: mốc thời gian (`start_ms`,
            // `slice_from_ms`) tính từ `dropped`, quên đếm là trục thời gian lệch
            // đúng bằng đoạn vừa cắt.
            self.dropped += 1;
        }
    }

    /// Mốc thời gian (ms) của sample NGAY SAU sample cuối trong buffer.
    pub fn end_ms(&self) -> i64 {
        self.start_ms() + (self.samples.len() as i64 * 1_000) / self.sample_rate as i64
    }

    pub fn clear(&mut self) {
        self.samples.clear();
        self.dropped = 0;
    }

    /// Audio từ mốc `start_ms` (tính từ đầu lượt nói) tới hết buffer. Mốc đã bị
    /// trôi khỏi buffer thì lấy từ đầu buffer.
    pub fn slice_from_ms(&self, start_ms: i64) -> Vec<f32> {
        let wanted = (self.sample_rate as i64 * start_ms.max(0) / 1_000) as u64;
        let skip = wanted.saturating_sub(self.dropped) as usize;
        if skip >= self.samples.len() {
            return Vec::new();
        }
        self.samples.iter().skip(skip).copied().collect()
    }

    /// Mốc thời gian (ms, tính từ đầu lượt nói) của sample đầu tiên còn trong buffer.
    pub fn start_ms(&self) -> i64 {
        (self.dropped * 1_000 / self.sample_rate as u64) as i64
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
    fn slice_from_ms_accounts_for_dropped_samples() {
        let mut buffer = AudioRingBuffer::new(16_000, 1.0);
        buffer.push(&vec![0.1; 16_000]);
        // Đẩy thêm 0,5 s -> 0,5 s đầu bị bỏ, buffer bắt đầu ở mốc 500 ms.
        buffer.push(&vec![0.2; 8_000]);
        assert_eq!(buffer.start_ms(), 500);
        // Lấy từ mốc 1000 ms -> đúng phần 0,5 s cuối.
        assert_eq!(buffer.slice_from_ms(1_000).len(), 8_000);
        // Mốc đã trôi mất -> lấy từ đầu buffer.
        assert_eq!(buffer.slice_from_ms(0).len(), 16_000);
        // Mốc vượt quá dữ liệu -> rỗng.
        assert!(buffer.slice_from_ms(5_000).is_empty());
    }

    #[test]
    fn retain_tail_advances_the_time_axis() {
        let mut buffer = AudioRingBuffer::new(16_000, 4.0);
        buffer.push(&vec![0.0; 32_000]); // 2 s
        buffer.retain_tail(0.5);
        assert_eq!(buffer.len(), 8_000);
        assert_eq!(buffer.start_ms(), 1_500);
        assert_eq!(buffer.end_ms(), 2_000);
        // Cắt xong, slice từ mốc cũ phải trả đúng phần còn lại chứ không lệch.
        assert_eq!(buffer.slice_from_ms(1_500).len(), 8_000);
    }

    #[test]
    fn tail_returns_only_the_requested_duration() {
        let mut buffer = AudioRingBuffer::new(16_000, 4.0);
        buffer.push(&vec![0.0; 48_000]);

        assert_eq!(buffer.tail(1.0).len(), 16_000);
        assert_eq!(buffer.tail(10.0).len(), buffer.len());
    }
}

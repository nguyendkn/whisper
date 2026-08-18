/// Gom text của một session.
///
/// Partial chỉ decode vài giây cuối, nên bản thân nó không phải cả bài. Client
/// muốn hiển thị liền mạch thì cần "phần đã chốt + đuôi đang chạy" — đó là việc
/// của struct này.
#[derive(Debug, Default)]
pub struct Transcript {
    committed: Vec<String>,
    duplicates_dropped: usize,
}

/// Kết quả của một lần chốt: `accepted = false` nghĩa là text bị loại, caller
/// không nên gửi event ra client.
#[derive(Debug, Clone)]
pub struct CommitOutcome {
    pub full_text: String,
    pub accepted: bool,
}

impl Transcript {
    pub fn new() -> Self {
        Self::default()
    }

    /// Chốt text của một lượt nói.
    ///
    /// Loại text rỗng và text **trùng y nguyên lượt trước**: whisper ảo giác trên
    /// khoảng lặng thường lặp lại cùng một câu học từ dữ liệu train, còn hai lượt
    /// nói thật liền nhau giống hệt nhau thì gần như không xảy ra.
    pub fn commit(&mut self, text: &str) -> CommitOutcome {
        let text = text.trim();
        if text.is_empty() {
            return CommitOutcome {
                full_text: self.committed_text(),
                accepted: false,
            };
        }
        if self.committed.last().map(String::as_str) == Some(text) {
            self.duplicates_dropped += 1;
            return CommitOutcome {
                full_text: self.committed_text(),
                accepted: false,
            };
        }
        self.committed.push(text.to_string());
        CommitOutcome {
            full_text: self.committed_text(),
            accepted: true,
        }
    }

    pub fn duplicates_dropped(&self) -> usize {
        self.duplicates_dropped
    }

    /// Toàn văn đã chốt cộng đuôi partial (không lưu lại partial).
    pub fn with_partial(&self, partial: &str) -> String {
        let partial = partial.trim();
        if partial.is_empty() {
            return self.committed.join(" ");
        }
        if self.committed.is_empty() {
            return partial.to_string();
        }
        format!("{} {}", self.committed.join(" "), partial)
    }

    pub fn committed_text(&self) -> String {
        self.committed.join(" ")
    }

    pub fn utterances(&self) -> usize {
        self.committed.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_then_partial_reads_as_one_stream() {
        let mut transcript = Transcript::new();
        let outcome = transcript.commit(" xin chào ");
        assert!(outcome.accepted);
        assert_eq!(outcome.full_text, "xin chào");
        assert_eq!(transcript.with_partial("hôm nay"), "xin chào hôm nay");
        assert_eq!(transcript.with_partial("   "), "xin chào");
        assert_eq!(transcript.utterances(), 1);
    }

    #[test]
    fn empty_final_is_not_committed() {
        let mut transcript = Transcript::new();
        let outcome = transcript.commit("  ");
        assert!(!outcome.accepted);
        assert_eq!(outcome.full_text, "");
        assert_eq!(transcript.utterances(), 0);
    }

    #[test]
    fn identical_consecutive_utterance_is_dropped() {
        let mut transcript = Transcript::new();
        assert!(transcript.commit("hãy đăng ký kênh").accepted);
        // Lượt thứ hai giống hệt -> ảo giác lặp, không chốt.
        assert!(!transcript.commit("hãy đăng ký kênh").accepted);
        assert_eq!(transcript.utterances(), 1);
        assert_eq!(transcript.duplicates_dropped(), 1);
        // Text khác thì vẫn nhận bình thường.
        assert!(transcript.commit("một câu khác").accepted);
        assert_eq!(transcript.utterances(), 2);
    }
}

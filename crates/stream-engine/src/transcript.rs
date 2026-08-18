/// Gom text của một session.
///
/// Partial chỉ decode vài giây cuối, nên bản thân nó không phải cả bài. Client
/// muốn hiển thị liền mạch thì cần "phần đã chốt + đuôi đang chạy" — đó là việc
/// của struct này.
#[derive(Debug, Default)]
pub struct Transcript {
    committed: Vec<String>,
}

impl Transcript {
    pub fn new() -> Self {
        Self::default()
    }

    /// Chốt text của một lượt nói, trả về toàn văn tới thời điểm này.
    pub fn commit(&mut self, text: &str) -> String {
        let text = text.trim();
        if !text.is_empty() {
            self.committed.push(text.to_string());
        }
        self.committed.join(" ")
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
        assert_eq!(transcript.commit(" xin chào "), "xin chào");
        assert_eq!(transcript.with_partial("hôm nay"), "xin chào hôm nay");
        assert_eq!(transcript.with_partial("   "), "xin chào");
        assert_eq!(transcript.utterances(), 1);
    }

    #[test]
    fn empty_final_is_not_committed() {
        let mut transcript = Transcript::new();
        assert_eq!(transcript.commit("  "), "");
        assert_eq!(transcript.utterances(), 0);
    }
}

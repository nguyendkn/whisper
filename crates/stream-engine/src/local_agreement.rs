//! Chính sách streaming LocalAgreement-2 (Liu et al. 2020, dùng trong
//! whisper_streaming của Macháček et al.): **chốt phần tiền tố chung dài nhất của
//! hai lần decode liên tiếp**.
//!
//! Vì sao cần: cách đơn giản là mỗi lượt partial decode lại cửa sổ đuôi rồi hiển thị
//! nguyên kết quả — text nhảy qua nhảy lại vì model đổi ý ở phần chưa có đủ ngữ cảnh.
//! LocalAgreement chỉ chốt những gì hai lượt liên tiếp đồng ý, nên phần đã hiện ra
//! không bị viết lại nữa, và phần audio đã chốt được cắt khỏi cửa sổ decode tiếp theo.
//!
//! AlignAtt (chính sách tốt hơn trong SimulStreaming) cần đọc cross-attention từng
//! bước decode — whisper.cpp không mở ra, nên không dùng được ở đây.

use whisper_core::Word;

/// Sai số cho phép khi so mốc thời gian giữa hai lượt decode.
const TIME_TOLERANCE_MS: i64 = 100;

#[derive(Debug, Default)]
pub struct LocalAgreement {
    committed: Vec<Word>,
    /// Hypothesis của lượt trước, phần chưa chốt.
    previous: Vec<Word>,
}

impl LocalAgreement {
    pub fn new() -> Self {
        Self::default()
    }

    /// Nạp hypothesis mới (mốc thời gian tuyệt đối trong lượt nói), trả về những từ
    /// vừa được chốt.
    pub fn insert(&mut self, hypothesis: Vec<Word>) -> Vec<Word> {
        let boundary = self.committed_end_ms();
        let fresh: Vec<Word> = hypothesis
            .into_iter()
            .filter(|word| word.start_ms + TIME_TOLERANCE_MS >= boundary)
            .collect();

        let mut newly = Vec::new();
        let mut index = 0;
        while index < fresh.len() && index < self.previous.len() {
            if !same_word(&fresh[index], &self.previous[index]) {
                break;
            }
            newly.push(fresh[index].clone());
            index += 1;
        }

        self.committed.extend(newly.iter().cloned());
        self.previous = fresh[index..].to_vec();
        newly
    }

    /// Mốc kết thúc của phần đã chốt — cửa sổ decode tiếp theo bắt đầu từ đây.
    pub fn committed_end_ms(&self) -> i64 {
        self.committed.last().map(|word| word.end_ms).unwrap_or(0)
    }

    pub fn committed_text(&self) -> String {
        join(&self.committed)
    }

    /// Phần chưa chốt của lượt gần nhất — hiện ra được nhưng có thể còn đổi.
    pub fn pending_text(&self) -> String {
        join(&self.previous)
    }

    pub fn is_empty(&self) -> bool {
        self.committed.is_empty() && self.previous.is_empty()
    }

    /// Trượt cửa sổ decode mà **không** bỏ phần đã chốt: chỉ hypothesis đang treo
    /// mất chỗ neo, còn text đã hiện ra cho người dùng thì không bao giờ bị rút lại.
    pub fn slide(&mut self) {
        self.previous.clear();
    }

    /// Xoá sạch khi đóng lượt nói.
    pub fn reset(&mut self) {
        self.committed.clear();
        self.previous.clear();
    }
}

fn same_word(left: &Word, right: &Word) -> bool {
    normalize(&left.text) == normalize(&right.text)
        && (left.start_ms - right.start_ms).abs() <= TIME_TOLERANCE_MS * 5
}

fn normalize(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn join(words: &[Word]) -> String {
    let mut out = String::new();
    for word in words {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word.text.trim());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start_ms: i64) -> Word {
        Word {
            text: text.to_string(),
            start_ms,
            end_ms: start_ms + 300,
        }
    }

    #[test]
    fn nothing_is_committed_from_the_first_hypothesis() {
        let mut agreement = LocalAgreement::new();
        let committed = agreement.insert(vec![word("xin", 0), word("chào", 400)]);
        assert!(committed.is_empty());
        assert_eq!(agreement.pending_text(), "xin chào");
        assert_eq!(agreement.committed_text(), "");
    }

    #[test]
    fn common_prefix_of_two_hypotheses_is_committed() {
        let mut agreement = LocalAgreement::new();
        agreement.insert(vec![word("xin", 0), word("chào", 400), word("bạm", 800)]);
        // Lượt sau đồng ý hai từ đầu, sửa từ thứ ba.
        let committed = agreement.insert(vec![word("xin", 0), word("chào", 400), word("bạn", 800)]);
        assert_eq!(committed.len(), 2);
        assert_eq!(agreement.committed_text(), "xin chào");
        assert_eq!(agreement.pending_text(), "bạn");
        assert_eq!(agreement.committed_end_ms(), 700);
    }

    #[test]
    fn words_before_the_committed_boundary_are_ignored() {
        let mut agreement = LocalAgreement::new();
        agreement.insert(vec![word("một", 1_000), word("hai", 1_400)]);
        agreement.insert(vec![word("một", 1_000), word("hai", 1_400)]);
        assert_eq!(agreement.committed_text(), "một hai");
        // Hypothesis mới lặp lại phần đã chốt -> bỏ, chỉ giữ phần sau mốc.
        agreement.insert(vec![
            word("một", 1_000),
            word("hai", 1_400),
            word("ba", 1_800),
        ]);
        assert_eq!(agreement.pending_text(), "ba");
    }

    #[test]
    fn punctuation_and_case_do_not_block_agreement() {
        let mut agreement = LocalAgreement::new();
        agreement.insert(vec![word("Hưng", 0), word("Yên,", 400)]);
        let committed = agreement.insert(vec![word("hưng", 0), word("yên", 400)]);
        assert_eq!(committed.len(), 2);
    }

    #[test]
    fn slide_keeps_committed_text() {
        let mut agreement = LocalAgreement::new();
        agreement.insert(vec![word("một", 0), word("hai", 400)]);
        agreement.insert(vec![word("một", 0), word("hai", 400)]);
        assert_eq!(agreement.committed_text(), "một hai");
        agreement.slide();
        // Phần đã chốt còn nguyên, chỉ hypothesis treo bị bỏ.
        assert_eq!(agreement.committed_text(), "một hai");
        assert_eq!(agreement.pending_text(), "");
    }

    #[test]
    fn reset_clears_everything() {
        let mut agreement = LocalAgreement::new();
        agreement.insert(vec![word("a", 0)]);
        agreement.insert(vec![word("a", 0)]);
        agreement.reset();
        assert!(agreement.is_empty());
        assert_eq!(agreement.committed_end_ms(), 0);
    }
}

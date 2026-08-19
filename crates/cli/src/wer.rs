//! Word Error Rate — thước đo khách quan để so các cấu hình decode với nhau.
//!
//! Chuẩn hoá trước khi so: chữ thường, bỏ dấu câu, gộp khoảng trắng. **Giữ nguyên
//! dấu tiếng Việt** — bỏ dấu sẽ che mất đúng loại lỗi mà ta cần đo.

/// Kết quả so một transcript với reference.
#[derive(Debug, Clone, PartialEq)]
pub struct WerReport {
    pub reference_words: usize,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
}

impl WerReport {
    pub fn errors(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }

    pub fn wer(&self) -> f32 {
        if self.reference_words == 0 {
            return if self.insertions == 0 { 0.0 } else { 1.0 };
        }
        self.errors() as f32 / self.reference_words as f32
    }
}

pub fn normalize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// Levenshtein trên chuỗi từ, truy vết để tách riêng thay/xoá/thêm.
pub fn compare(reference: &str, hypothesis: &str) -> WerReport {
    let reference = normalize(reference);
    let hypothesis = normalize(hypothesis);
    let (n, m) = (reference.len(), hypothesis.len());

    // cost[i][j] = số lỗi ít nhất để biến ref[..i] thành hyp[..j].
    let mut cost = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in cost.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in cost[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = cost[i - 1][j - 1] + usize::from(reference[i - 1] != hypothesis[j - 1]);
            cost[i][j] = sub.min(cost[i - 1][j] + 1).min(cost[i][j - 1] + 1);
        }
    }

    let (mut i, mut j) = (n, m);
    let mut report = WerReport {
        reference_words: n,
        substitutions: 0,
        deletions: 0,
        insertions: 0,
    };
    while i > 0 || j > 0 {
        if i > 0
            && j > 0
            && reference[i - 1] == hypothesis[j - 1]
            && cost[i][j] == cost[i - 1][j - 1]
        {
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && cost[i][j] == cost[i - 1][j - 1] + 1 {
            report.substitutions += 1;
            i -= 1;
            j -= 1;
        } else if i > 0 && cost[i][j] == cost[i - 1][j] + 1 {
            report.deletions += 1;
            i -= 1;
        } else {
            report.insertions += 1;
            j -= 1;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_has_no_errors() {
        let report = compare("Xin chào, thế giới!", "xin chào thế giới");
        assert_eq!(report.errors(), 0);
        assert_eq!(report.wer(), 0.0);
    }

    #[test]
    fn counts_each_error_type() {
        // ref: a b c d | hyp: a x c d e  -> 1 thay (b->x), 1 thêm (e)
        let report = compare("a b c d", "a x c d e");
        assert_eq!(report.substitutions, 1);
        assert_eq!(report.insertions, 1);
        assert_eq!(report.deletions, 0);
        assert_eq!(report.reference_words, 4);
        assert!((report.wer() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn missing_words_count_as_deletions() {
        let report = compare("một hai ba bốn", "một ba bốn");
        assert_eq!(report.deletions, 1);
        assert_eq!(report.substitutions, 0);
    }

    #[test]
    fn diacritics_are_significant() {
        // "hao" và "hào" phải bị tính là khác nhau.
        assert_eq!(compare("mỹ hào", "my hao").errors(), 2);
    }
}

//! Guard cho index.html: mọi `els.<tên>` mà JS dùng phải nằm trong `ELS_IDS` và
//! phải có phần tử `id="<tên>"` trong HTML.
//!
//! Đây đúng là lớp lỗi đã xảy ra thật: thêm `els.level.textContent` mà quên khai
//! báo — handler audio chết im lặng ngay frame đầu, người dùng thấy
//! "mic không gửi được audio". Không có browser trong CI nên guard bằng text.

use std::collections::BTreeSet;

const HTML: &str = include_str!("../assets/index.html");

fn els_ids_list() -> BTreeSet<String> {
    let start = HTML.find("const ELS_IDS = [").expect("thiếu ELS_IDS");
    let end = HTML[start..].find("];").expect("ELS_IDS không đóng") + start;
    HTML[start..end]
        .split('\'')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect()
}

fn els_usages() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let bytes = HTML.as_bytes();
    let mut at = 0;
    while let Some(pos) = HTML[at..].find("els.") {
        let word_start = at + pos + 4;
        let word: String = HTML[word_start..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !word.is_empty() && bytes[at + pos].is_ascii() {
            out.insert(word);
        }
        at = word_start;
    }
    // `els[id]` trong vòng khởi tạo không phải usage tĩnh.
    out
}

#[test]
fn every_els_usage_is_declared_and_has_a_dom_element() {
    let declared = els_ids_list();
    assert!(!declared.is_empty());

    for name in els_usages() {
        assert!(
            declared.contains(&name),
            "JS dùng els.{name} nhưng ELS_IDS không khai báo — chính lớp lỗi \
             'mic không gửi được audio' đã gặp"
        );
    }
    for id in &declared {
        assert!(
            HTML.contains(&format!("id=\"{id}\"")),
            "ELS_IDS khai báo '{id}' nhưng HTML không có phần tử id=\"{id}\""
        );
    }
}

#[test]
fn audio_worklet_stays_connected_to_the_destination() {
    // Bất biến thứ hai từng vỡ: worklet không nằm trên đường tới destination thì
    // Chrome không gọi process() và không byte nào được gửi.
    assert!(HTML.contains("node.connect(silence).connect(audioCtx.destination)"));
}

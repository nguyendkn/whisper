//! Backend Zipformer RNN-T chạy qua sherpa-onnx (offline recognizer).
//!
//! Dùng cho các model k2/icefall xuất ONNX (encoder/decoder/joiner + tokens.txt),
//! ví dụ `hynt/Zipformer-30M-RNNT-6000h` cho tiếng Việt. Khác whisper ở mọi điểm
//! vận hành: không giới hạn input >= 1 s, không pad lên 30 s, RTF trên CPU nhỏ hơn
//! cả trăm lần — nhưng chỉ một ngôn ngữ và không có chấm câu/hoa thường.

use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sherpa_rs::sherpa_rs_sys as sys;
use whisper_core::{
    AsrBackend, AsrError, DecodeMode, Segment, TranscriptResult, Word, WHISPER_SAMPLE_RATE,
};

/// Ngắn hơn mức này thì trả kết quả rỗng thay vì gọi model — RNN-T decode được
/// audio cực ngắn nhưng vài chục ms thì chỉ có noise.
const MIN_AUDIO_MS: u32 = 120;

pub struct ZipformerConfig {
    /// Thư mục chứa encoder/decoder/joiner `.onnx` và `tokens.txt`.
    pub dir: PathBuf,
    /// Dùng bản int8 (nhanh hơn, nhẹ hơn; đổi một ít độ chính xác).
    pub quantized: bool,
    pub n_threads: i32,
}

pub struct ZipformerBackend {
    recognizer: *const sys::SherpaOnnxOfflineRecognizer,
    n_threads: usize,
}

// Recognizer của sherpa-onnx dùng được từ nhiều thread, mỗi lượt decode tạo
// stream riêng (giống WhisperContext + WhisperState).
unsafe impl Send for ZipformerBackend {}
unsafe impl Sync for ZipformerBackend {}

impl Drop for ZipformerBackend {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyOfflineRecognizer(self.recognizer) }
    }
}

/// Tìm file theo vai trò trong thư mục model: chứa `role` + đuôi `.onnx`, đúng
/// biến thể int8 hay không.
fn find_model_file(dir: &Path, role: &str, quantized: bool) -> Result<PathBuf, AsrError> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| AsrError::Backend(format!("đọc {}: {e}", dir.display())))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            name.contains(role) && name.ends_with(".onnx") && name.contains(".int8.") == quantized
        })
        .collect();
    candidates.sort();
    candidates.into_iter().next().ok_or_else(|| {
        AsrError::Backend(format!(
            "không thấy file {role}{} trong {}",
            if quantized { " (int8)" } else { "" },
            dir.display()
        ))
    })
}

impl ZipformerBackend {
    pub fn load(config: ZipformerConfig) -> Result<Self, AsrError> {
        let tokens = config.dir.join("tokens.txt");
        if !tokens.exists() {
            return Err(AsrError::Backend(format!(
                "thiếu {} (với model của hynt: đổi tên config.json thành tokens.txt)",
                tokens.display()
            )));
        }
        let encoder = find_model_file(&config.dir, "encoder", config.quantized)?;
        let decoder = find_model_file(&config.dir, "decoder", config.quantized)?;
        let joiner = find_model_file(&config.dir, "joiner", config.quantized)?;

        let cstr = |path: &Path| {
            CString::new(path.to_string_lossy().as_bytes())
                .map_err(|e| AsrError::Backend(e.to_string()))
        };
        let encoder_c = cstr(&encoder)?;
        let decoder_c = cstr(&decoder)?;
        let joiner_c = cstr(&joiner)?;
        let tokens_c = cstr(&tokens)?;
        let provider_c = CString::new("cpu").expect("static");
        let decoding_c = CString::new("greedy_search").expect("static");

        // Zero-init rồi chỉ set field mình cần: struct của sherpa-onnx thêm field
        // mới theo version, liệt kê đủ là tự gãy khi nâng cấp.
        let started = std::time::Instant::now();
        let recognizer = unsafe {
            let mut model_config: sys::SherpaOnnxOfflineModelConfig = std::mem::zeroed();
            model_config.transducer = sys::SherpaOnnxOfflineTransducerModelConfig {
                encoder: encoder_c.as_ptr(),
                decoder: decoder_c.as_ptr(),
                joiner: joiner_c.as_ptr(),
            };
            model_config.tokens = tokens_c.as_ptr();
            model_config.provider = provider_c.as_ptr();
            model_config.num_threads = config.n_threads.max(1);

            let mut recognizer_config: sys::SherpaOnnxOfflineRecognizerConfig = std::mem::zeroed();
            recognizer_config.model_config = model_config;
            recognizer_config.decoding_method = decoding_c.as_ptr();

            sys::SherpaOnnxCreateOfflineRecognizer(&recognizer_config)
        };
        if recognizer.is_null() {
            return Err(AsrError::Backend(format!(
                "sherpa-onnx không tạo được recognizer từ {}",
                config.dir.display()
            )));
        }

        tracing::info!(
            dir = %config.dir.display(),
            quantized = config.quantized,
            n_threads = config.n_threads,
            load_ms = started.elapsed().as_millis() as u64,
            "zipformer model loaded"
        );

        Ok(Self {
            recognizer,
            n_threads: config.n_threads.max(1) as usize,
        })
    }
}

#[derive(Deserialize)]
struct SherpaResult {
    #[serde(default)]
    text: String,
    #[serde(default)]
    tokens: Vec<String>,
    #[serde(default)]
    timestamps: Vec<f32>,
}

/// Ghép token BPE thành từ: token bắt đầu bằng `▁` (hoặc khoảng trắng) mở từ mới.
fn words_from_tokens(tokens: &[String], timestamps: &[f32]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        let ts_ms = timestamps
            .get(index)
            .map(|t| (t * 1_000.0) as i64)
            .unwrap_or(0);
        let starts_word = token.starts_with('▁') || token.starts_with(' ') || words.is_empty();
        let piece = token.trim_start_matches(['▁', ' ']).to_lowercase();
        if piece.is_empty() {
            continue;
        }
        if starts_word {
            words.push(Word {
                text: piece,
                start_ms: ts_ms,
                // RNN-T chỉ cho mốc bắt đầu của token; mốc kết thúc lấy từ token
                // sau (vá lại ở vòng dưới), tạm +200 ms cho token cuối.
                end_ms: ts_ms + 200,
            });
        } else if let Some(last) = words.last_mut() {
            last.text.push_str(&piece);
            last.end_ms = last.end_ms.max(ts_ms + 200);
        }
    }
    // Mốc kết thúc của một từ = mốc bắt đầu của từ kế tiếp (RNN-T không phát mốc
    // kết thúc); từ cuối giữ ước lượng +200 ms ở trên.
    for i in 0..words.len().saturating_sub(1) {
        let next_start = words[i + 1].start_ms;
        if next_start > words[i].start_ms {
            words[i].end_ms = next_start;
        }
    }
    words
}

impl AsrBackend for ZipformerBackend {
    fn transcribe(
        &self,
        pcm: &[f32],
        mode: DecodeMode,
        _prompt: Option<&str>,
        _language: Option<&str>,
    ) -> Result<TranscriptResult, AsrError> {
        let audio_ms = (pcm.len() as u64 * 1_000 / WHISPER_SAMPLE_RATE as u64) as u32;
        if audio_ms < MIN_AUDIO_MS {
            return Err(AsrError::AudioTooShort {
                got_ms: audio_ms,
                min_ms: MIN_AUDIO_MS,
            });
        }

        let started = std::time::Instant::now();
        let parsed: SherpaResult = unsafe {
            let stream = sys::SherpaOnnxCreateOfflineStream(self.recognizer);
            sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                WHISPER_SAMPLE_RATE as i32,
                pcm.as_ptr(),
                pcm.len() as i32,
            );
            sys::SherpaOnnxDecodeOfflineStream(self.recognizer, stream);
            let result = sys::SherpaOnnxGetOfflineStreamResult(stream);
            // Đọc qua field json thay vì layout struct: json ổn định giữa các
            // version sherpa-onnx, layout thì không.
            let json = if result.is_null() || (*result).json.is_null() {
                String::from("{}")
            } else {
                CStr::from_ptr((*result).json)
                    .to_string_lossy()
                    .into_owned()
            };
            sys::SherpaOnnxDestroyOfflineRecognizerResult(result);
            sys::SherpaOnnxDestroyOfflineStream(stream);
            serde_json::from_str(&json)
                .map_err(|e| AsrError::Backend(format!("parse kết quả sherpa: {e}")))?
        };
        let inference_ms = started.elapsed().as_millis() as u64;

        // Model của hynt phát token IN HOA — hạ về chữ thường cho UI/WER.
        let text = parsed.text.trim().to_lowercase();
        let words = words_from_tokens(&parsed.tokens, &parsed.timestamps);
        let end_ms = words
            .last()
            .map(|word| word.end_ms)
            .unwrap_or(audio_ms as i64);

        Ok(TranscriptResult {
            segments: if text.is_empty() {
                Vec::new()
            } else {
                vec![Segment {
                    text,
                    start_ms: words.first().map(|word| word.start_ms).unwrap_or(0),
                    end_ms,
                }]
            },
            words,
            mode,
            audio_ms,
            inference_ms,
        })
    }

    fn n_threads(&self) -> usize {
        self.n_threads
    }

    fn name(&self) -> &'static str {
        "zipformer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpe_tokens_become_words_with_timestamps() {
        let tokens: Vec<String> = ["▁XIN", "▁CH", "ÀO", "▁BẠN"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let timestamps = vec![0.0, 0.4, 0.56, 0.8];
        let words = words_from_tokens(&tokens, &timestamps);
        assert_eq!(
            words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            vec!["xin", "chào", "bạn"]
        );
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[0].end_ms, 400); // kết thúc = mốc bắt đầu từ sau
        assert_eq!(words[1].start_ms, 400);
    }
}

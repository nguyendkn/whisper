//! Integration test chốt hành vi transcribe khi đổi model/backend.
//!
//! Cần dữ liệu thật (không dùng audio giả): trỏ hai biến môi trường tới model và
//! một file WAV 16 kHz mono có lời nói, kèm một từ chắc chắn phải xuất hiện.
//!
//!   WHISPER_RT_TEST_MODEL=models/ggml-large-v3-turbo.bin \
//!   WHISPER_RT_TEST_WAV=tests/data/hello.wav \
//!   WHISPER_RT_TEST_EXPECT="xin chào" \
//!   cargo test -p core --test transcribe_wav -- --nocapture
//!
//! Thiếu biến thì test tự bỏ qua (in ra lý do) để CI không cần tải model 1,5 GB.

use std::path::PathBuf;

use whisper_core::{transcribe, DecodeMode, WhisperConfig, WhisperModel, WHISPER_SAMPLE_RATE};

#[test]
fn transcribes_a_real_wav_file() {
    let (Ok(model_path), Ok(wav_path)) = (
        std::env::var("WHISPER_RT_TEST_MODEL"),
        std::env::var("WHISPER_RT_TEST_WAV"),
    ) else {
        eprintln!("bỏ qua: cần WHISPER_RT_TEST_MODEL và WHISPER_RT_TEST_WAV");
        return;
    };

    let config = WhisperConfig {
        model_path: PathBuf::from(model_path),
        language: std::env::var("WHISPER_RT_TEST_LANG")
            .ok()
            .or(Some("vi".into())),
        ..WhisperConfig::default()
    };
    let model = WhisperModel::load(config).expect("load model");
    let mut state = model.create_state().expect("create state");

    let pcm = read_wav_mono_16k(&PathBuf::from(wav_path));
    let result = transcribe(&model, &mut state, &pcm, DecodeMode::Final, None)
        .expect("transcribe should succeed");

    let text = result.text().to_lowercase();
    eprintln!("rtf={:.2} text={text}", result.rtf());
    assert!(!text.is_empty(), "transcript rỗng");
    if let Ok(expected) = std::env::var("WHISPER_RT_TEST_EXPECT") {
        assert!(
            text.contains(&expected.to_lowercase()),
            "không thấy {expected:?} trong {text:?}"
        );
    }
}

fn read_wav_mono_16k(path: &PathBuf) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("mở file WAV");
    let spec = reader.spec();
    assert_eq!(
        spec.sample_rate, WHISPER_SAMPLE_RATE,
        "test WAV phải là 16 kHz"
    );
    assert_eq!(spec.channels, 1, "test WAV phải là mono");
    match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.unwrap() as f32 / scale)
                .collect()
        }
    }
}

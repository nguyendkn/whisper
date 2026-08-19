//! End-to-end WebSocket: chạy binary server thật, stream một file WAV lên,
//! nhận partial/final.
//!
//! Cần dữ liệu thật, không dùng audio giả:
//!
//!   WHISPER_RT_TEST_MODEL=models/ggml-tiny.en.bin \
//!   WHISPER_RT_TEST_WAV=samples/jfk.wav \
//!   WHISPER_RT_TEST_LANG=en WHISPER_RT_TEST_EXPECT=country \
//!   cargo test -p server --test ws_stream -- --nocapture
//!
//! Đặt `WHISPER_RT_TEST_URL=wss://host/v1/stream` để bắn vào một server đang chạy
//! sẵn (ví dụ bản deploy thật) thay vì tự spawn binary — dùng để tách lỗi phía server
//! khỏi lỗi phía browser.
//!
//! Thiếu biến thì test tự bỏ qua.

use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;

const PORT: u16 = 18_123;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn streams_a_wav_file_and_receives_a_final_transcript() {
    let (Ok(model), Ok(wav)) = (
        std::env::var("WHISPER_RT_TEST_MODEL"),
        std::env::var("WHISPER_RT_TEST_WAV"),
    ) else {
        eprintln!("bỏ qua: cần WHISPER_RT_TEST_MODEL và WHISPER_RT_TEST_WAV");
        return;
    };

    let remote = std::env::var("WHISPER_RT_TEST_URL").ok();
    let mut server = match remote.as_deref() {
        Some(url) => {
            eprintln!("dùng server sẵn có: {url}");
            None
        }
        None => {
            let child = spawn_server(&model);
            wait_until_listening().await;
            Some(child)
        }
    };

    let url = remote
        .unwrap_or_else(|| format!("ws://127.0.0.1:{PORT}/v1/stream?sample_rate=16000&channels=1"));
    let (mut socket, _) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("kết nối websocket");

    let samples = read_wav_i16_16k_mono(&PathBuf::from(wav));
    // 1 600 sample = 100 ms; gửi nhanh hơn thời gian thực để test không kéo dài.
    for chunk in samples.chunks(1_600) {
        let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
        socket
            .send(Message::Binary(bytes.into()))
            .await
            .expect("gửi audio");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    socket
        .send(Message::text(r#"{"type":"eos"}"#))
        .await
        .expect("gửi eos");

    let expected = std::env::var("WHISPER_RT_TEST_EXPECT").unwrap_or_else(|_| "country".into());
    let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
    let mut saw_ready = false;
    let mut finals: Vec<String> = Vec::new();

    while tokio::time::Instant::now() < deadline {
        let Ok(Some(Ok(message))) =
            tokio::time::timeout(Duration::from_secs(30), socket.next()).await
        else {
            break;
        };
        let Ok(text) = message.into_text() else {
            continue;
        };
        let Ok(payload) = serde_json::from_str::<serde_json::Value>(text.as_str()) else {
            continue;
        };
        match payload["type"].as_str() {
            Some("ready") => saw_ready = true,
            Some("partial") => eprintln!("partial: {}", payload["text"]),
            Some("final") => {
                eprintln!("final: {}", payload["text"]);
                finals.push(payload["text"].as_str().unwrap_or_default().to_lowercase());
            }
            Some("error") => panic!("server báo lỗi: {payload}"),
            _ => {}
        }
        if finals.iter().any(|text| text.contains(&expected)) {
            break;
        }
    }

    if let Some(server) = server.as_mut() {
        let _ = server.kill();
        let _ = server.wait();
    }
    assert!(saw_ready, "không nhận được frame ready");
    assert!(
        finals.iter().any(|text| text.contains(&expected)),
        "không thấy {expected:?} trong các final: {finals:?}"
    );
}

fn spawn_server(model: &str) -> Child {
    let mut command = Command::new(env!("CARGO_BIN_EXE_whisper-rt-server"));
    command
        // Config nằm ở gốc workspace, còn CWD của test là thư mục crate.
        .env("WHISPER_RT_CONFIG", "../../config/default")
        .env("WHISPER_RT__BIND_ADDR", format!("127.0.0.1:{PORT}"))
        .env("WHISPER_RT__MODEL__PATH", model)
        .env("WHISPER_RT__MODEL__USE_GPU", "false")
        .env(
            "WHISPER_RT__MODEL__LANGUAGE",
            std::env::var("WHISPER_RT_TEST_LANG").unwrap_or_else(|_| "en".into()),
        )
        .env(
            "WHISPER_RT__VAD__MODEL_PATH",
            std::env::var("WHISPER_RT_TEST_VAD_MODEL").unwrap_or_default(),
        );
    command.spawn().expect("chạy được binary server")
}

async fn wait_until_listening() {
    for _ in 0..100 {
        if TcpStream::connect(("127.0.0.1", PORT)).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("server không mở port {PORT}");
}

fn read_wav_i16_16k_mono(path: &PathBuf) -> Vec<i16> {
    let mut reader = hound::WavReader::open(path).expect("mở WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "test WAV phải 16 kHz");
    assert_eq!(spec.channels, 1, "test WAV phải mono");
    match spec.sample_format {
        hound::SampleFormat::Int => reader.samples::<i16>().map(Result::unwrap).collect(),
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| (sample.unwrap() * i16::MAX as f32) as i16)
            .collect(),
    }
}

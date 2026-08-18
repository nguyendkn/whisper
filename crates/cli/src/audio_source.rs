use std::path::Path;
use std::time::Duration;

use audio_pipeline::{decode_file_to_16k_mono, AudioResampler, MicCapture, TARGET_SAMPLE_RATE};
use stream_engine::Session;

/// Kích thước chunk khi phát lại file: 100 ms giống nhịp một client streaming.
const FILE_CHUNK_MS: usize = 100;

/// Đọc mic tới khi Ctrl+C, đẩy từng chunk đã resample vào session.
pub async fn run_microphone(session: &mut Session, queue_len: usize) -> anyhow::Result<()> {
    let (capture, mut rx) = MicCapture::start(queue_len)?;
    let mut resampler = AudioResampler::new(capture.sample_rate(), capture.channels() as usize)?;
    println!(
        "listening on {} ({} Hz, {} ch) — Ctrl+C để dừng",
        capture.device_name(),
        capture.sample_rate(),
        capture.channels()
    );

    loop {
        tokio::select! {
            chunk = rx.recv() => match chunk {
                Some(chunk) => {
                    let pcm = resampler.push(&chunk)?;
                    session.push_pcm(&pcm);
                }
                None => break,
            },
            _ = tokio::signal::ctrl_c() => {
                println!("\nđang chốt phần cuối...");
                break;
            }
        }
    }

    let tail = resampler.flush()?;
    session.push_pcm(&tail);
    Ok(())
}

/// Phát một file audio (mp3, wav, flac, ogg, m4a...) vào session. `realtime` mô phỏng đúng nhịp thời gian thực
/// để kiểm tra VAD và partial; tắt đi thì nạp nhanh nhất có thể (chỉ dùng để
/// kiểm tra chất lượng text, không phản ánh độ trễ).
pub async fn run_file(session: &mut Session, path: &Path, realtime: bool) -> anyhow::Result<()> {
    let pcm = decode_file_to_16k_mono(path)?;
    let chunk_samples = TARGET_SAMPLE_RATE as usize * FILE_CHUNK_MS / 1_000;
    println!(
        "phát {} ({:.1} s, realtime={realtime})",
        path.display(),
        pcm.len() as f32 / TARGET_SAMPLE_RATE as f32
    );

    for chunk in pcm.chunks(chunk_samples) {
        session.push_pcm(chunk);
        if realtime {
            tokio::time::sleep(Duration::from_millis(FILE_CHUNK_MS as u64)).await;
        } else {
            // Nhường runtime để các task inference chạy được.
            tokio::task::yield_now().await;
        }
    }
    Ok(())
}

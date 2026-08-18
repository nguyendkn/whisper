//! Giải mã file/byte audio (mp3, wav, flac, ogg, m4a...) thành PCM f32 16 kHz
//! mono — định dạng duy nhất mà whisper nhận.
//!
//! Dùng symphonia (pure Rust) thay vì gọi ffmpeg: không thêm dependency hệ
//! thống, deploy chỉ cần một binary.

use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use symphonia::core::codecs::CodecParameters;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::TrackType;
use symphonia::core::io::MediaSourceStream;

use crate::{error::AudioError, resampler::AudioResampler};

/// Giải mã một file trên đĩa.
pub fn decode_file_to_16k_mono(path: &Path) -> Result<Vec<f32>, AudioError> {
    let file =
        File::open(path).map_err(|e| AudioError::Decode(format!("mở {}: {e}", path.display())))?;
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    decode(stream, &hint)
}

/// Giải mã audio đã nằm trong bộ nhớ (body của một HTTP request chẳng hạn).
/// Định dạng được nhận diện bằng nội dung, không cần đuôi file.
pub fn decode_bytes_to_16k_mono(bytes: Vec<u8>) -> Result<Vec<f32>, AudioError> {
    let stream = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    decode(stream, &Hint::new())
}

fn decode(stream: MediaSourceStream<'_>, hint: &Hint) -> Result<Vec<f32>, AudioError> {
    let mut format = symphonia::default::get_probe()
        .probe(hint, stream, Default::default(), Default::default())
        .map_err(|e| AudioError::Decode(format!("không nhận diện được định dạng: {e}")))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| AudioError::Decode("file không có audio track".into()))?;
    let track_id = track.id;
    let Some(CodecParameters::Audio(params)) = track.codec_params.clone() else {
        return Err(AudioError::Decode("thiếu codec parameters".into()));
    };

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &Default::default())
        .map_err(|e| AudioError::Decode(format!("không có decoder cho codec: {e}")))?;

    let mut resampler: Option<AudioResampler> = None;
    let mut interleaved: Vec<f32> = Vec::new();
    let mut pcm: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            // Stream cắt giữa (file tải dở) — dùng phần đã giải mã được.
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(err) => return Err(AudioError::Decode(err.to_string())),
        };
        if packet.track_id != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // Packet lỗi giữa file: bỏ packet đó, giữ phần còn lại.
            Err(SymphoniaError::DecodeError(err)) => {
                tracing::debug!(%err, "bỏ packet lỗi");
                continue;
            }
            Err(err) => return Err(AudioError::Decode(err.to_string())),
        };

        let rate = decoded.spec().rate();
        let channels = decoded.spec().channels().count().max(1);
        interleaved.clear();
        decoded.copy_to_vec_interleaved(&mut interleaved);

        let resampler = match resampler.as_mut() {
            Some(resampler) => resampler,
            None => {
                tracing::debug!(rate, channels, "audio source format");
                resampler.insert(AudioResampler::new(rate, channels)?)
            }
        };
        pcm.extend(resampler.push(&interleaved)?);
    }

    if let Some(resampler) = resampler.as_mut() {
        pcm.extend(resampler.flush()?);
    }
    if pcm.is_empty() {
        return Err(AudioError::Decode("không giải mã được sample nào".into()));
    }
    Ok(pcm)
}

//! Capture mic qua cpal — chỉ dùng cho `cli`. Server nhận audio từ client qua
//! WebSocket nên không cần module này (feature `capture` mặc định tắt ở server).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, FromSample, SampleFormat, SizedSample, Stream, StreamConfig};
use tokio::sync::mpsc;

use crate::error::AudioError;

/// Handle giữ stream sống. Drop handle = dừng capture; đừng `mem::forget`
/// stream, vì khi đó không còn cách nào tắt mic.
pub struct MicCapture {
    stream: Stream,
    sample_rate: u32,
    channels: u16,
    device_name: String,
}

impl MicCapture {
    /// Mở input device mặc định ở format gốc của nó. Sample rate/channels trả
    /// về gần như chắc chắn **không** phải 16 kHz mono — nối tiếp qua
    /// [`crate::AudioResampler`] trước khi đưa vào ASR.
    pub fn start(queue_len: usize) -> Result<(Self, mpsc::Receiver<Vec<f32>>), AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(AudioError::NoInputDevice)?;
        let device_name = device
            .description()
            .map(|description| description.name().to_string())
            .unwrap_or_else(|_| "<unknown>".into());
        let supported = device
            .default_input_config()
            .map_err(|e| AudioError::Device(e.to_string()))?;

        let sample_format = supported.sample_format();
        let sample_rate = supported.sample_rate();
        let channels = supported.channels();
        let config = supported.config();

        let (tx, rx) = mpsc::channel::<Vec<f32>>(queue_len.max(1));
        let stream = match sample_format {
            SampleFormat::F32 => build_stream::<f32>(&device, &config, tx),
            SampleFormat::F64 => build_stream::<f64>(&device, &config, tx),
            SampleFormat::I8 => build_stream::<i8>(&device, &config, tx),
            SampleFormat::I16 => build_stream::<i16>(&device, &config, tx),
            SampleFormat::I32 => build_stream::<i32>(&device, &config, tx),
            SampleFormat::U8 => build_stream::<u8>(&device, &config, tx),
            SampleFormat::U16 => build_stream::<u16>(&device, &config, tx),
            SampleFormat::U32 => build_stream::<u32>(&device, &config, tx),
            other => Err(AudioError::UnsupportedSampleFormat(other.to_string())),
        }?;

        stream
            .play()
            .map_err(|e| AudioError::Device(e.to_string()))?;
        tracing::info!(
            device = %device_name,
            sample_rate,
            channels,
            format = %sample_format,
            "mic capture started"
        );

        Ok((
            Self {
                stream,
                sample_rate,
                channels,
                device_name,
            },
            rx,
        ))
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn pause(&self) -> Result<(), AudioError> {
        self.stream
            .pause()
            .map_err(|e| AudioError::Device(e.to_string()))
    }
}

fn build_stream<T>(
    device: &Device,
    config: &StreamConfig,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<Stream, AudioError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    device
        .build_input_stream::<T, _, _>(
            *config,
            move |data: &[T], _| {
                let pcm: Vec<f32> = data.iter().copied().map(f32::from_sample_).collect();
                // Callback của cpal chạy trên thread realtime: không được block.
                // Queue đầy nghĩa là consumer không theo kịp — bỏ chunk và log.
                if tx.try_send(pcm).is_err() {
                    tracing::warn!("capture queue full, dropping audio chunk");
                }
            },
            |err| tracing::error!(%err, "cpal input stream error"),
            None,
        )
        .map_err(|e| AudioError::Device(e.to_string()))
}

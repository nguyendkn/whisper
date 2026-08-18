use std::path::PathBuf;

#[derive(thiserror::Error, Debug)]
pub enum AudioError {
    #[error("no input device found")]
    NoInputDevice,
    #[error("audio device error: {0}")]
    Device(String),
    #[error("unsupported sample format: {0}")]
    UnsupportedSampleFormat(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("resampler error: {0}")]
    Resample(String),
    #[error("VAD model not found: {0}")]
    VadModelMissing(PathBuf),
    #[error("VAD error: {0}")]
    Vad(String),
    #[error("frame of {got} samples does not match VAD frame size {expected}")]
    VadFrameSize { got: usize, expected: usize },
}

use std::path::Path;

use whisper_rs::{WhisperVadContext, WhisperVadContextParams};

use crate::{error::AudioError, vad::SpeechProbe};

/// Silero chấm điểm theo frame 512 sample (32 ms ở 16 kHz).
pub const SILERO_FRAME_SAMPLES: usize = 512;

/// Silero VAD do whisper.cpp bundle sẵn (model GGML riêng, không phải model ASR).
///
/// Dùng đường này thay vì `onnxruntime`: cùng một runtime GGML đã link, không
/// thêm dependency native nào.
///
/// Lưu ý: whisper.cpp reset state nội bộ mỗi lượt `detect_speech`, nên hãy nạp
/// từng đoạn >= vài trăm ms thay vì từng frame 32 ms để LSTM có đủ ngữ cảnh.
pub struct SileroVad {
    ctx: WhisperVadContext,
}

impl SileroVad {
    pub fn load(model_path: &Path, n_threads: i32, use_gpu: bool) -> Result<Self, AudioError> {
        if !model_path.exists() {
            return Err(AudioError::VadModelMissing(model_path.to_path_buf()));
        }
        let mut params = WhisperVadContextParams::new();
        params.set_n_threads(n_threads);
        params.set_use_gpu(use_gpu);

        let path = model_path.to_string_lossy().into_owned();
        let ctx = WhisperVadContext::new(&path, params)
            .map_err(|e| AudioError::Vad(format!("load {}: {e}", model_path.display())))?;

        tracing::info!(model = %model_path.display(), "silero VAD loaded");
        Ok(Self { ctx })
    }
}

impl SpeechProbe for SileroVad {
    fn frame_samples(&self) -> usize {
        SILERO_FRAME_SAMPLES
    }

    fn probabilities(&mut self, pcm: &[f32]) -> Result<Vec<f32>, AudioError> {
        if pcm.len() < SILERO_FRAME_SAMPLES {
            return Err(AudioError::VadFrameSize {
                got: pcm.len(),
                expected: SILERO_FRAME_SAMPLES,
            });
        }
        self.ctx
            .detect_speech(pcm)
            .map_err(|e| AudioError::Vad(e.to_string()))?;
        Ok(self.ctx.probabilities().to_vec())
    }
}

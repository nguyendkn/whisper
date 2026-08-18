use whisper_rs::{WhisperContext, WhisperContextParameters, WhisperState};

use crate::{config::WhisperConfig, error::AsrError};

/// Bọc `WhisperContext` — nặng (weights), load một lần rồi share qua `Arc`.
///
/// Context là `Send + Sync`; mỗi lượt inference cần một `WhisperState` riêng.
/// `WhisperState` giữ `Arc` tới context nên không có lifetime ràng buộc, dùng
/// được qua `spawn_blocking` — xem [`crate::StatePool`].
#[derive(Debug)]
pub struct WhisperModel {
    ctx: WhisperContext,
    config: WhisperConfig,
}

impl WhisperModel {
    pub fn load(config: WhisperConfig) -> Result<Self, AsrError> {
        if !config.model_path.exists() {
            return Err(AsrError::ModelMissing(config.model_path.clone()));
        }

        let params = WhisperContextParameters {
            use_gpu: config.use_gpu,
            gpu_device: config.gpu_device,
            flash_attn: config.flash_attn,
            ..Default::default()
        };

        let started = std::time::Instant::now();
        let ctx =
            WhisperContext::new_with_params(&config.model_path, params).map_err(|source| {
                AsrError::ModelLoad {
                    path: config.model_path.clone(),
                    source,
                }
            })?;

        tracing::info!(
            model = %config.model_path.display(),
            use_gpu = config.use_gpu,
            multilingual = ctx.is_multilingual(),
            load_ms = started.elapsed().as_millis() as u64,
            "whisper model loaded"
        );

        Ok(Self { ctx, config })
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    /// Cấp phát một state mới. **Không rẻ**: whisper.cpp cấp KV cache và buffer
    /// theo kích thước model (hàng trăm MB với large-v3), nên đừng gọi mỗi
    /// chunk — mượn qua [`crate::StatePool`].
    pub fn create_state(&self) -> Result<WhisperState, AsrError> {
        self.ctx.create_state().map_err(AsrError::StateCreate)
    }
}

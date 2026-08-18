use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use audio_pipeline::{AudioRingBuffer, SpeechGate, SpeechProbe, VadEvent, TARGET_SAMPLE_RATE};
use tokio::sync::mpsc;
use uuid::Uuid;
use whisper_core::{AsrError, DecodeMode};

use crate::{
    config::SessionConfig,
    error::EngineError,
    event::{StreamEvent, TranscriptUpdate},
    scheduler::InferenceScheduler,
    transcript::Transcript,
};

/// Model dùng cho một session.
///
/// Cho phép **hai model khác nhau**: một model nhỏ/nhanh cho partial và model lớn
/// cho final — cách RealtimeSTT gọi là `realtime_model_type`. Đây là cách duy nhất
/// để dùng `large-v3-turbo` mà partial vẫn dưới 300 ms: đo trên CPU 16 core, turbo
/// cần ~2 s cho cửa sổ partial 6 s, còn `base` chỉ ~250 ms.
#[derive(Clone)]
pub struct SessionEngines {
    /// Model chốt câu — chất lượng quyết định ở đây.
    pub finals: Arc<InferenceScheduler>,
    /// Model chạy partial. `None` = dùng luôn model final.
    pub partials: Option<Arc<InferenceScheduler>>,
}

impl SessionEngines {
    pub fn single(scheduler: Arc<InferenceScheduler>) -> Self {
        Self {
            finals: scheduler,
            partials: None,
        }
    }

    fn for_mode(&self, mode: DecodeMode) -> &Arc<InferenceScheduler> {
        match mode {
            DecodeMode::Partial => self.partials.as_ref().unwrap_or(&self.finals),
            DecodeMode::Final => &self.finals,
        }
    }
}

/// Một phiên transcribe realtime (một kết nối WebSocket, hoặc một lần chạy CLI).
///
/// Không sở hữu model: mọi lượt inference đi qua [`InferenceScheduler`] để cả
/// hệ thống dùng chung một bản weights và một hạn mức concurrency.
pub struct Session {
    id: Uuid,
    config: SessionConfig,
    engines: SessionEngines,
    events: mpsc::Sender<StreamEvent>,
    buffer: AudioRingBuffer,
    gate: SpeechGate,
    probe: Box<dyn SpeechProbe>,
    probe_pending: Vec<f32>,
    probe_chunk_samples: usize,
    max_probe_backlog: usize,
    transcript: Arc<Mutex<Transcript>>,
    partial_inflight: Arc<AtomicBool>,
    last_partial_at: Instant,
    utterance: u64,
}

impl Session {
    pub fn new(
        engines: SessionEngines,
        probe: Box<dyn SpeechProbe>,
        events: mpsc::Sender<StreamEvent>,
        config: SessionConfig,
    ) -> Self {
        // Nạp VAD theo cụm ~8 frame (~256 ms): đủ ngữ cảnh cho Silero, vẫn đủ
        // mịn để cắt câu.
        let probe_chunk_samples = probe.frame_samples() * 8;
        let frame_ms = probe.frame_ms();
        Self {
            id: Uuid::new_v4(),
            gate: SpeechGate::new(config.gate, frame_ms),
            buffer: AudioRingBuffer::new(TARGET_SAMPLE_RATE, config.max_utterance_secs),
            max_probe_backlog: (TARGET_SAMPLE_RATE as f32 * config.max_probe_backlog_secs) as usize,
            config,
            engines,
            events,
            probe,
            probe_pending: Vec::with_capacity(probe_chunk_samples * 2),
            probe_chunk_samples,
            transcript: Arc::new(Mutex::new(Transcript::new())),
            partial_inflight: Arc::new(AtomicBool::new(false)),
            last_partial_at: Instant::now(),
            utterance: 0,
        }
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn committed_text(&self) -> String {
        self.transcript
            .lock()
            .expect("transcript poisoned")
            .committed_text()
    }

    /// Nạp một chunk PCM f32 16 kHz mono. Hàm này **không** chờ inference: mọi
    /// lượt decode được `spawn` ra task riêng, nên vòng đọc audio của caller
    /// không bao giờ bị chặn. Cần chạy trong tokio runtime.
    pub fn push_pcm(&mut self, pcm: &[f32]) {
        if pcm.is_empty() {
            return;
        }
        self.buffer.push(pcm);
        self.probe_pending.extend_from_slice(pcm);

        // VAD không theo kịp: bỏ phần chờ cũ nhất. Audio vẫn nằm trong ring buffer
        // nên nội dung không mất, chỉ mất độ chính xác của mốc cắt câu.
        if self.probe_pending.len() > self.max_probe_backlog {
            let drop_len = self.probe_pending.len() - self.max_probe_backlog;
            self.probe_pending.drain(..drop_len);
            tracing::warn!(
                session_id = %self.id,
                dropped_samples = drop_len,
                "VAD chậm hơn luồng audio vào, bỏ phần chờ cũ nhất"
            );
        }

        while self.probe_pending.len() >= self.probe_chunk_samples {
            let chunk: Vec<f32> = self
                .probe_pending
                .drain(..self.probe_chunk_samples)
                .collect();
            let probs = match self.probe.probabilities(&chunk) {
                Ok(probs) => probs,
                Err(err) => {
                    tracing::warn!(session_id = %self.id, %err, "VAD failed on chunk");
                    continue;
                }
            };
            for prob in probs {
                let event = self.gate.observe(prob);
                self.handle_vad_event(event);
            }
        }

        if self.buffer.duration_secs() >= self.config.max_utterance_secs {
            tracing::debug!(session_id = %self.id, "utterance hit the length cap, closing it");
            self.gate.force_end();
            self.close_utterance();
        }
    }

    /// Chốt phần còn lại (client đóng kết nối, hoặc hết file).
    pub fn finish(&mut self) {
        self.gate.force_end();
        self.close_utterance();
    }

    fn handle_vad_event(&mut self, event: VadEvent) {
        match event {
            VadEvent::SpeechStart => {
                // Bỏ khoảng lặng đứng trước, chỉ giữ pre-roll.
                self.buffer.retain_tail(self.config.pre_roll_secs);
                self.last_partial_at = Instant::now();
            }
            VadEvent::SpeechContinue => self.maybe_partial(),
            VadEvent::SpeechEnd => self.close_utterance(),
            VadEvent::Silence => {}
        }
    }

    fn maybe_partial(&mut self) {
        if self.last_partial_at.elapsed().as_millis() < self.config.partial_interval_ms as u128 {
            return;
        }
        // Lượt partial trước chưa xong: bỏ lượt này thay vì xếp hàng chồng lên
        // nhau, nếu không queue sẽ phình mãi khi RTF > 1.
        if self.partial_inflight.swap(true, Ordering::AcqRel) {
            tracing::trace!(session_id = %self.id, "skipping partial, previous one still running");
            return;
        }
        self.last_partial_at = Instant::now();
        let pcm = self.buffer.tail(self.config.partial_window_secs);
        self.spawn_decode(pcm, DecodeMode::Partial, self.utterance);
    }

    fn close_utterance(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let pcm = self.buffer.snapshot();
        self.buffer.clear();
        let utterance = self.utterance;
        self.utterance += 1;
        self.spawn_decode(pcm, DecodeMode::Final, utterance);
    }

    fn spawn_decode(&self, pcm: Vec<f32>, mode: DecodeMode, utterance: u64) {
        let scheduler = Arc::clone(self.engines.for_mode(mode));
        let events = self.events.clone();
        let transcript = Arc::clone(&self.transcript);
        let inflight = Arc::clone(&self.partial_inflight);
        let session_id = self.id;

        tokio::spawn(async move {
            let outcome = scheduler.submit(pcm, mode).await;
            if mode == DecodeMode::Partial {
                inflight.store(false, Ordering::Release);
            }

            let event = match outcome {
                Ok(result) => {
                    let text = result.text();
                    let full_text = match mode {
                        DecodeMode::Final => {
                            let outcome = transcript
                                .lock()
                                .expect("transcript poisoned")
                                .commit(&text);
                            if !outcome.accepted {
                                tracing::debug!(
                                    session_id = %session_id,
                                    utterance,
                                    "bỏ final rỗng hoặc trùng lượt trước"
                                );
                                return;
                            }
                            outcome.full_text
                        }
                        DecodeMode::Partial => transcript
                            .lock()
                            .expect("transcript poisoned")
                            .with_partial(&text),
                    };
                    let update = TranscriptUpdate {
                        session_id,
                        utterance,
                        text,
                        full_text,
                        audio_ms: result.audio_ms,
                        rtf: result.rtf(),
                    };
                    match mode {
                        DecodeMode::Final => StreamEvent::Final(update),
                        DecodeMode::Partial => StreamEvent::Partial(update),
                    }
                }
                // Audio ngắn hơn ngưỡng tối thiểu không phải lỗi của client —
                // im lặng bỏ qua thay vì bắn error ra socket.
                Err(EngineError::Asr(AsrError::AudioTooShort { got_ms, min_ms })) => {
                    tracing::debug!(session_id = %session_id, got_ms, min_ms, "skipped short chunk");
                    return;
                }
                Err(err) => {
                    tracing::error!(session_id = %session_id, %err, ?mode, "inference failed");
                    StreamEvent::Error {
                        session_id,
                        message: err.to_string(),
                    }
                }
            };

            if events.send(event).await.is_err() {
                tracing::debug!(session_id = %session_id, "event receiver gone, dropping result");
            }
        });
    }
}

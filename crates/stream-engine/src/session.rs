use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use audio_pipeline::{AudioRingBuffer, SpeechGate, SpeechProbe, VadEvent, TARGET_SAMPLE_RATE};
use tokio::sync::mpsc;
use uuid::Uuid;
use whisper_core::{AsrError, DecodeMode};

/// Log tình trạng session sau mỗi bấy nhiêu **giây audio** (không phải giây treo):
/// với luồng realtime hai cái tương đương, còn khi nạp nhanh (test, chạy file) thì
/// mốc theo thời gian treo sẽ không bao giờ tới và telemetry thành vô dụng.
const HEARTBEAT_AUDIO_SECS: f32 = 5.0;

use crate::{
    config::SessionConfig,
    error::EngineError,
    event::{StreamEvent, TranscriptUpdate},
    local_agreement::LocalAgreement,
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
    agreement: Arc<Mutex<LocalAgreement>>,
    /// Mốc (ms trong lượt nói) mà cửa sổ partial tiếp theo bắt đầu — LocalAgreement
    /// đẩy mốc này lên mỗi khi chốt thêm từ.
    partial_start_ms: Arc<AtomicI64>,
    partial_inflight: Arc<AtomicBool>,
    last_partial_at: Instant,
    utterance: u64,
    /// Nhịp log định kỳ: khi người dùng báo "không thấy transcript", ba con số
    /// samples/biên độ/xác suất VAD nói ngay lỗi nằm ở đâu — không có audio, audio
    /// im lặng, hay VAD không mở.
    last_heartbeat_at: Instant,
    samples_since_heartbeat: usize,
    peak_since_heartbeat: f32,
    max_prob_since_heartbeat: f32,
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
            agreement: Arc::new(Mutex::new(LocalAgreement::new())),
            partial_start_ms: Arc::new(AtomicI64::new(0)),
            partial_inflight: Arc::new(AtomicBool::new(false)),
            last_partial_at: Instant::now(),
            utterance: 0,
            last_heartbeat_at: Instant::now(),
            samples_since_heartbeat: 0,
            peak_since_heartbeat: 0.0,
            max_prob_since_heartbeat: 0.0,
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
        self.samples_since_heartbeat += pcm.len();
        self.peak_since_heartbeat = pcm.iter().fold(self.peak_since_heartbeat, |peak, sample| {
            peak.max(sample.abs())
        });

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
                self.max_prob_since_heartbeat = self.max_prob_since_heartbeat.max(prob);
                let event = self.gate.observe(prob);
                self.handle_vad_event(event);
            }
        }

        self.maybe_heartbeat();

        if self.buffer.duration_secs() >= self.config.max_utterance_secs {
            tracing::debug!(session_id = %self.id, "utterance hit the length cap, closing it");
            self.gate.force_end();
            self.close_utterance();
        }
    }

    /// Log tình trạng luồng vào sau mỗi `HEARTBEAT_AUDIO_SECS` giây audio.
    fn maybe_heartbeat(&mut self) {
        let audio_secs = self.samples_since_heartbeat as f32 / TARGET_SAMPLE_RATE as f32;
        if audio_secs < HEARTBEAT_AUDIO_SECS {
            return;
        }
        tracing::info!(
            session_id = %self.id,
            audio_secs,
            wall_secs = self.last_heartbeat_at.elapsed().as_secs_f32(),
            peak_amplitude = self.peak_since_heartbeat,
            max_speech_prob = self.max_prob_since_heartbeat,
            speaking = self.gate.is_speaking(),
            buffered_secs = self.buffer.duration_secs(),
            utterances = self.utterance,
            "session heartbeat"
        );
        self.last_heartbeat_at = Instant::now();
        self.samples_since_heartbeat = 0;
        self.peak_since_heartbeat = 0.0;
        self.max_prob_since_heartbeat = 0.0;
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

        let (pcm, window_start_ms) = self.partial_window();
        if pcm.is_empty() {
            self.partial_inflight.store(false, Ordering::Release);
            return;
        }
        self.spawn_decode(
            pcm,
            DecodeMode::Partial,
            self.utterance,
            None,
            window_start_ms,
        );
    }

    /// Cửa sổ audio cho lượt partial tiếp theo, kèm mốc bắt đầu của nó.
    ///
    /// Có LocalAgreement thì cửa sổ bắt đầu từ chỗ đã chốt — decode ít audio hơn và
    /// không lặp lại phần đã xong. Không có thì lấy đuôi cố định như cũ.
    fn partial_window(&mut self) -> (Vec<f32>, i64) {
        if !self.config.local_agreement {
            return (self.buffer.tail(self.config.partial_window_secs), -1);
        }

        let max_samples = (TARGET_SAMPLE_RATE as f32 * self.config.partial_window_secs) as usize;
        let mut start_ms = self
            .partial_start_ms
            .load(Ordering::Acquire)
            .max(self.buffer.start_ms());
        let mut pcm = self.buffer.slice_from_ms(start_ms);

        // Không chốt được gì trong một cửa sổ đầy (nhạc nền, người nói lấp bấp):
        // trượt cửa sổ về đuôi và bỏ giao kèo cũ, nếu không cửa sổ phình mãi.
        if pcm.len() > max_samples {
            pcm = self.buffer.tail(self.config.partial_window_secs);
            start_ms = self.buffer.duration_secs() as i64 * 1_000 + self.buffer.start_ms()
                - (self.config.partial_window_secs * 1_000.0) as i64;
            self.partial_start_ms
                .store(start_ms.max(0), Ordering::Release);
            self.agreement.lock().expect("agreement poisoned").slide();
            tracing::debug!(session_id = %self.id, start_ms, "trượt cửa sổ partial, giữ phần đã chốt");
        }
        (pcm, start_ms.max(0))
    }

    fn close_utterance(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let pcm = self.buffer.snapshot();
        self.buffer.clear();
        let utterance = self.utterance;
        self.utterance += 1;
        let prompt = self.previous_text_prompt();
        self.agreement.lock().expect("agreement poisoned").reset();
        self.partial_start_ms.store(0, Ordering::Release);
        self.spawn_decode(pcm, DecodeMode::Final, utterance, prompt, 0);
    }

    /// Đuôi text đã chốt, dùng mồi cho lượt Final tiếp theo.
    fn previous_text_prompt(&self) -> Option<String> {
        if !self.config.condition_on_previous {
            return None;
        }
        let committed = self
            .transcript
            .lock()
            .expect("transcript poisoned")
            .committed_text();
        if committed.is_empty() {
            return None;
        }
        let start = committed
            .char_indices()
            .rev()
            .take(self.config.prompt_chars)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        Some(committed[start..].to_string())
    }

    fn spawn_decode(
        &self,
        pcm: Vec<f32>,
        mode: DecodeMode,
        utterance: u64,
        prompt: Option<String>,
        window_start_ms: i64,
    ) {
        let scheduler = Arc::clone(self.engines.for_mode(mode));
        let events = self.events.clone();
        let transcript = Arc::clone(&self.transcript);
        let inflight = Arc::clone(&self.partial_inflight);
        let agreement = Arc::clone(&self.agreement);
        let partial_start_ms = Arc::clone(&self.partial_start_ms);
        let use_agreement = self.config.local_agreement && window_start_ms >= 0;
        let language = self.config.language.clone();
        let session_id = self.id;

        tokio::spawn(async move {
            let outcome = scheduler.submit(pcm, mode, prompt, language).await;
            if mode == DecodeMode::Partial {
                inflight.store(false, Ordering::Release);
            }

            let event = match outcome {
                Ok(result) => {
                    let mut text = result.text();
                    let mut stable_text = String::new();
                    // LocalAgreement: chỉ hiện phần hai lượt liên tiếp đồng ý, và
                    // đẩy mốc bắt đầu của cửa sổ sau lên chỗ đã chốt.
                    if use_agreement && mode == DecodeMode::Partial {
                        let words = result
                            .words
                            .iter()
                            .map(|word| whisper_core::Word {
                                text: word.text.clone(),
                                start_ms: word.start_ms + window_start_ms,
                                end_ms: word.end_ms + window_start_ms,
                            })
                            .collect();
                        let mut agreement = agreement.lock().expect("agreement poisoned");
                        agreement.insert(words);
                        let committed = agreement.committed_text();
                        let pending = agreement.pending_text();
                        partial_start_ms.store(agreement.committed_end_ms(), Ordering::Release);
                        drop(agreement);
                        text = match (committed.is_empty(), pending.is_empty()) {
                            (true, _) => pending.clone(),
                            (false, true) => committed.clone(),
                            (false, false) => format!("{committed} {pending}"),
                        };
                        stable_text = committed;
                        if text.trim().is_empty() {
                            return;
                        }
                    }
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
                        stable_text,
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

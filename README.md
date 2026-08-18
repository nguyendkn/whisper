# whisper-rt

Streaming speech-to-text realtime bằng Rust: whisper.cpp qua `whisper-rs`, VAD Silero
(bundle sẵn trong whisper.cpp — không cần onnxruntime), WebSocket + REST qua axum.

Chạy được cả trên GPU lớn (nhiều session song song) và trên NUC (ít tài nguyên,
giới hạn concurrency ở `stream-engine`).

## Cấu trúc

```
crates/core/            # PCM f32 16 kHz mono -> text. Không biết mic/WebSocket/session.
crates/audio-pipeline/  # capture (cpal), resample về 16 kHz mono (rubato), VAD, ring buffer.
crates/stream-engine/   # session, partial/final, StatePool + Semaphore giới hạn inference.
crates/server/          # axum: WS /v1/stream, POST /v1/transcribe, GET /health.
crates/cli/             # mic hoặc file WAV -> terminal.
```

Ranh giới quan trọng: `core` không biết gì về audio realtime, `audio-pipeline` không
biết gì về ASR, `server` chỉ lo giao thức. Đổi model (Voxtral, Parakeet) chỉ đụng
`core`; đổi axum sang framework khác chỉ đụng `server`.

Crate `core` có `[lib] name = "whisper_core"` — tên package là `core` theo layout, còn
tên lib phải khác để không mờ nghĩa với `core` của Rust.

## Chuẩn bị

```bash
./scripts/download_model.sh                # large-v3-turbo + Silero VAD
./scripts/download_model.sh tiny.en        # model nhỏ để thử nhanh
```

Yêu cầu build: Rust 1.87+, cmake, C++ compiler, libclang (bindgen). Trên Linux thêm
`libasound2-dev` nếu build `cli` (feature `capture`).

Backend tăng tốc chọn bằng feature, mặc định là CPU:

```bash
cargo build --release -p server --features cuda      # NVIDIA
cargo build --release -p server --features vulkan    # iGPU/NPU qua Vulkan
cargo build --release -p server --features openblas  # CPU + BLAS
```

## Chạy

```bash
# CLI: mic
cargo run --release -p cli -- --model models/ggml-large-v3-turbo.bin \
    --vad-model models/ggml-silero-v5.1.2.bin --language vi

# CLI: file WAV, mô phỏng đúng nhịp thời gian thực
cargo run --release -p cli -- --file mau.wav --model models/ggml-tiny.en.bin --language en

# Server
cargo run --release -p server
```

Config: `config/default.toml`. Override bằng env (`WHISPER_RT__MODEL__PATH=...`,
hai gạch dưới tách theo tầng) hoặc đổi file bằng `WHISPER_RT_CONFIG`.

## API

`GET /health` → trạng thái, version whisper.cpp, số permit còn rảnh.

`POST /v1/transcribe` — body là file WAV (int hoặc float, sample rate/channels nào
cũng được), trả toàn văn + segment:

```bash
curl --data-binary @mau.wav http://127.0.0.1:8080/v1/transcribe
```

`GET /v1/stream?sample_rate=48000&channels=1` — WebSocket:

- client gửi **binary frame** = PCM i16 little-endian ở sample rate đã khai báo
  (mặc định 16 kHz mono; khai báo khác thì server tự resample);
- client gửi `{"type":"eos"}` để chốt câu cuối;
- server trả text frame JSON: `ready`, `partial`, `final`, `error`.

```json
{"type":"final","utterance":0,"text":"...","full_text":"...","audio_ms":2400,"rtf":0.19}
```

`utterance` để bỏ partial về muộn hơn final của cùng lượt nói; `full_text` là các
lượt đã chốt cộng đuôi hiện tại.

## Những chỗ quyết định hiệu năng

- **whisper.cpp luôn pad input lên 30 s** trước encoder: decode 2 s tốn gần bằng
  decode 30 s. Vì vậy partial chỉ decode `session.partial_window_secs` giây cuối chứ
  không decode lại cả lượt nói, và chunk ngắn hơn `model.min_audio_ms` bị bỏ.
- **Partial không xếp hàng**: lượt trước chưa xong thì lượt mới bị bỏ, nên queue
  không phình khi RTF > 1.
- **`max_concurrent_inference` × `model.n_threads` ≤ số core vật lý.** Vượt là mọi
  session cùng chậm.
- **`StatePool`** giữ sẵn `WhisperState` (KV cache hàng trăm MB với large-v3) thay vì
  cấp phát lại mỗi chunk.
- **VAD cắt câu**: `vad.silence_ms_for_end` quyết định chỗ cắt. Nói chậm/nhiều nhịp
  nghỉ mà để thấp thì câu bị cắt giữa cụm từ.
- Log `rtf` theo từng chunk: `rtf` gần 1 nghĩa là sắp không theo kịp realtime.

## Test

```bash
cargo test --workspace          # unit test: resampler, VAD gate, ring buffer, transcript

# Test cần dữ liệu thật (tự bỏ qua nếu thiếu biến môi trường)
WHISPER_RT_TEST_MODEL=models/ggml-tiny.en.bin \
WHISPER_RT_TEST_WAV=samples/jfk.wav \
WHISPER_RT_TEST_LANG=en WHISPER_RT_TEST_EXPECT=country \
cargo test --workspace -- --nocapture
```

`crates/core/tests/transcribe_wav.rs` chốt hành vi transcribe khi đổi model/backend;
`crates/server/tests/ws_stream.rs` chạy binary server thật và stream WAV qua WebSocket.

## Chưa làm

- Metrics Prometheus (RTF, queue depth, p50/p95) — hiện chỉ có log `tracing`.
- Nhiều model song song trên một GPU (model nhỏ cho partial, model lớn cho final).
- LocalAgreement để ghép partial thay vì decode lại cửa sổ đuôi.
- Backpressure policy khi channel event đầy (hiện bounded 64, `send` chờ).

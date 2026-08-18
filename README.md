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

`POST /v1/transcribe` — body là file audio (mp3, wav, flac, ogg, m4a — sample rate
và số kênh nào cũng được, giải mã bằng symphonia nên không cần ffmpeg), trả toàn
văn + segment:

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

## Đo và tối ưu hiệu năng

`crates/cli` có sẵn chế độ benchmark; `scripts/bench_sweep.sh` quét tham số và in bảng:

```bash
./scripts/bench_sweep.sh models/ggml-base.bin samples/mau.mp3
# quét riêng throughput với tổng thread cố định:
SKIP_PHASE1=1 CPU_BUDGET=12 CONCURRENCY_LIST="1 2 3 4 6" \
  ./scripts/bench_sweep.sh models/ggml-base.bin samples/mau.mp3
# ghim vào một nhóm core (CPU hybrid):
PIN=0-5 ./scripts/bench_sweep.sh models/ggml-tiny.bin samples/mau.mp3
```

Ba con số nó đo: độ trễ partial (người dùng cảm nhận), độ trễ final, và throughput
tổng khi N stream chạy song song (`streams_at_rtf1` = số session một máy gánh được).

### Kết quả đo trên Intel Core Ultra 9 285H (16 core hybrid: 6 P + 8 E + 2 LP-E, CPU-only)

Model `base`, đoạn 20 s, cửa sổ partial 6 s:

| threads | streams | partial | final | streams @ RTF=1 |
|---|---|---|---|---|
| 8 | 1 | 257 ms | 1128 ms | 18.8 |
| 12 | 1 | 249 ms | 1107 ms | 18.0 |
| 4 | 1 | 317 ms | 1460 ms | 13.1 |
| 4 | 3 | 317 ms | 1326 ms | **25.0** |

Model `large-v3-turbo` (809M), cùng máy, CPU-only, đoạn 20 s:

| threads | streams | partial (6 s) | final (20 s) | streams @ RTF=1 |
|---|---|---|---|---|
| 12 | 1 | 1 985 ms (RTF 0,33) | 10 159 ms (RTF 0,51) | 2,0 |
| 8 | 1 | 2 110 ms (RTF 0,35) | 12 124 ms (RTF 0,61) | 1,8 |
| 6 | 2 | 3 148 ms | 15 550 ms | 2,0 |
| 4 | 1 | 3 881 ms | 20 532 ms (**RTF 1,03**) | 1,1 |

Đọc bảng này trước khi chọn model: turbo trên CPU-only chỉ gánh được **~2 stream**,
và với 4 thread thì **không kịp realtime cho một stream** (RTF > 1) — tức mặc định
`n_threads = 4, max_concurrent_inference = 3` là dành cho `base`/`small`. Dùng turbo
thì hoặc chuyển sang GPU (feature `cuda`/`vulkan`), hoặc đặt
`n_threads = 12, max_concurrent_inference = 1` và chấp nhận partial ~2 s.

Với turbo, `scale_partial_audio_ctx` còn quan trọng hơn nữa: **4,3×**
(8 593 ms → 1 985 ms cho cửa sổ 6 s). Tắt nó thì partial có RTF 1,43 — vô dụng.

Bốn kết luận rút ra từ vòng đo, đều đã đưa vào mặc định của `config/default.toml`:

1. **`scale_partial_audio_ctx` là khoản lãi lớn nhất của đường partial**: thu nhỏ
   encoder context theo đúng độ dài cửa sổ làm partial nhanh **2,3×**
   (`tiny`, 8 thread: 292 ms → 129 ms). Final vẫn chạy full context để không mất
   chất lượng cuối câu.
2. **Đừng bao giờ đặt `n_threads` bằng số core.** Trên máy này 14 thread/14 core
   chậm gấp 6 lần 12 thread, 16 thread chậm gấp 18 lần 8 thread. Chừa ≥ 2 core.
3. **Oversubscribe không xuống dốc từ từ mà sập**: 4 stream × 8 thread cho RTF
   2,19 (không kịp realtime) trong khi 2 stream × 8 thread cho 0,033. Giữ
   `streams × threads ≤ số core − 2`.
4. **Feature `openmp` chậm hơn** trên CPU hybrid này (1,3–1,9× ở mọi mức thread),
   và ghim riêng vào P-core cũng chậm hơn (mất 8 E-core). Để mặc định.

Với cùng tổng số thread, chia thành nhiều stream ít thread cho throughput cao hơn
(+33%) nhưng độ trễ mỗi stream cao hơn — chọn theo mục tiêu: hội thoại realtime thì
ưu tiên độ trễ, xử lý hàng loạt thì ưu tiên throughput.

### Ảo giác trên khoảng lặng

Whisper sinh text ngay cả khi đoạn audio không có tiếng nói — thường là câu học từ
dữ liệu train (kiểu lời chào kênh YouTube). Đo trên mẫu 128 s bằng turbo:

- `no_speech_probability` của segment **không dùng được** để phát hiện: ≈ 1e-5 cho
  mọi segment, kể cả segment ảo giác. Bộ lọc `no_speech_thold` vẫn giữ vì miễn phí
  và có tác dụng trên audio khác, nhưng đừng trông vào nó.
- Độ tự tin trung bình của token thì phân tách được: câu thật **0,936–0,977**, ảo
  giác **0,836–0,845**. Một câu thật rất ngắn cũng rơi xuống 0,82 — nên ngưỡng
  `min_confidence` mặc định **tắt** (0.0); đặt 0,88 sẽ dọn sạch ảo giác nhưng mất
  những câu ngắn nhất.
- Mặc định đang bật: **loại utterance trùng y nguyên lượt trước** — ảo giác kiểu này
  lặp lại cùng một câu, trong khi hai lượt nói thật liền nhau giống hệt nhau gần như
  không xảy ra. Trên mẫu trên, cách này bỏ được lần lặp thứ hai mà không cần ngưỡng.

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

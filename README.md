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

# CLI: model lớn chốt câu + model nhỏ cho partial (xem mục RealtimeSTT bên dưới)
cargo run --release -p cli -- --file mau.mp3 --language vi \
    --model models/ggml-large-v3-turbo.bin --partial-model models/ggml-base.bin
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

`GET /v1/stream?sample_rate=48000&channels=1&language=vi` — WebSocket
(`language` chọn model đã khai báo trong `[[language_models]]`, bỏ trống thì dùng model chính):

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
và với 4 thread thì **không kịp realtime cho một stream** (RTF > 1). Vì vậy mặc định
là `n_threads = 12, max_concurrent_inference = 1` — cấu hình độ trễ thấp nhất và là
cấu hình duy nhất chạy được turbo trên CPU (partial ~2 s). Cần nhiều session thì hoặc
chuyển sang GPU (feature `cuda`/`vulkan`), hoặc dùng `base`/`small` rồi chia thành
nhiều stream ít thread (4 thread × 3 stream, xem bảng trên).

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
ưu tiên độ trễ (mặc định), xử lý hàng loạt thì ưu tiên throughput.

Mặc định `n_threads = 12` được đo trên máy 16 core. Trên máy khác hãy đặt lại theo
đúng bất biến `streams × threads ≤ số core − 2`, đừng giữ nguyên 12.

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

## Nghiên cứu tham chiếu và thực nghiệm độ chính xác

### Nguồn

- **whisper_streaming** ([ufal](https://github.com/ufal/whisper_streaming), Macháček et al.,
  IJCNLP-AACL 2023 demo, [arXiv 2307.14743](https://arxiv.org/abs/2307.14743)) — chính sách
  **LocalAgreement-2**: chốt tiền tố chung dài nhất của hai lần decode liên tiếp, khử trùng
  n-gram khi ghép, cắt buffer theo ranh giới câu/segment.
- **SimulStreaming** ([ufal](https://github.com/ufal/SimulStreaming)) — chính sách **AlignAtt**
  (tốt hơn LocalAgreement) dùng cross-attention để quyết định khi nào dừng decode.
  **Không cài được ở đây**: whisper.cpp không mở cross-attention từng bước decode ra.
- **RealtimeSTT** ([KoljaB](https://github.com/KoljaB/RealtimeSTT)) — xem mục dưới.

`crates/stream-engine/src/local_agreement.rs` cài LocalAgreement-2 dùng mốc thời gian theo
token (`WhisperConfig::token_timestamps`, chỉ bật cho Partial). Cửa sổ partial bắt đầu từ mốc
đã chốt (`AudioRingBuffer::slice_from_ms`) nên không decode lại phần đã xong. Event partial trả
thêm `stable_text` — phần đã chốt, client render chữ chắc; phần còn lại render chữ mờ.

### WER thật trên FLEURS: chọn model theo ngôn ngữ

Bộ eval: [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google, CC-BY 4.0), 30 clip mỗi
thứ tiếng — tiếng Anh 590 từ / 257 s, tiếng Việt 831 từ / 341 s. Tải bằng
`./scripts/fetch_eval_set.sh`, chạy bằng `whisper-rt --eval-manifest eval/vi.tsv`. WER tính trên
toàn corpus (tổng lỗi / tổng từ), 12 thread, CPU-only.

**Tiếng Anh**

| Model | Cấu hình | WER | RTF |
|---|---|---|---|
| large-v3 | beam 5 | **3,90%** | 1,26 |
| large-v3-turbo | greedy | 4,58% | 0,99 |
| large-v3-turbo | beam 5 | 4,75% | 0,97 |
| large-v3-turbo | temp fallback | 4,75% | 1,01 |
| base | beam 5 | 8,14% | 0,06 |

**Tiếng Việt**

| Model | Cấu hình | WER | RTF |
|---|---|---|---|
| large-v3-turbo | beam 5 | **9,39%** | 0,79 |
| large-v3-turbo | temp fallback | 9,39% | 0,80 |
| large-v3-turbo | greedy | 10,23% | 0,79 |
| large-v3 | beam 5 | 11,43% | 1,11 |
| PhoWhisper-medium | beam 5 | 14,32% | 0,60 |
| PhoWhisper-small | beam 5 | 16,37% | 0,20 |
| PhoWhisper-medium | greedy | 25,99% | 0,56 |
| base | beam 5 | 42,24% | 0,07 |

Ba kết luận:

1. **Không có model nào thắng cả hai thứ tiếng.** `large-v3` hơn `turbo` 18% tương đối ở tiếng Anh
   nhưng kém 21% tương đối ở tiếng Việt. Vì thế server có `[[language_models]]`: khai báo model
   riêng cho từng ngôn ngữ, session chọn qua `?language=`. Muốn chính xác nhất cho cả hai thì chạy
   `en → large-v3`, `vi → turbo` (đổi lại RAM cho cả hai bản weights).
2. **Beam 5 giúp tiếng Việt (10,23% → 9,39%) nhưng là nước rửa với tiếng Anh** (4,58% vs 4,75% —
   lệch một lỗi trên 590 từ). Giữ mặc định beam 5 vì nó còn chống mất nội dung ở utterance dài
   (xem mục dưới), thứ mà bộ eval clip ngắn này không đo được.
3. **Fine-tune tiếng Việt không tự động thắng.** Hai bản ggml của PhoWhisper (VinAI, cộng đồng
   convert) đều tệ hơn turbo trên FLEURS, và lỗi tập trung ở **insertion** (52–54 lần) — dấu hiệu
   khác biệt quy ước chuẩn hoá text hơn là nghe sai. Kết quả này chỉ nói về FLEURS + cách chuẩn hoá
   của tôi; muốn phán xét công bằng cần đo thêm trên VIVOS/VLSP là những bộ mà PhoWhisper nhắm tới.

Hạn chế cần biết khi đọc bảng: 30 clip cho độ chính xác khoảng ±2 điểm phần trăm, nên chênh lệch
dưới ~2 điểm (như beam vs greedy ở tiếng Anh) là nhiễu. Số viết bằng chữ so với viết bằng số cũng
bị tính là lỗi vì tôi không chuẩn hoá số — điều này thổi WER lên đều nhau ở mọi cấu hình nên không
ảnh hưởng phép so sánh. `large-v3` có RTF 1,26 nên **không kịp realtime trên CPU máy này**: muốn
độ chính xác đó cho tiếng Anh thì cần GPU, hoặc dùng nó cho lượt Final trong khi partial chạy model nhỏ.

### Thực nghiệm: cấu hình decode nào thực sự làm tăng độ chính xác

Phương pháp (`scripts/experiment_accuracy.sh`, `crates/cli/src/wer.rs`): dựng tham chiếu
**oracle** bằng một lượt decode offline chất lượng cao, rồi đo WER của từng cấu hình streaming
so với nó — cô lập phần chất lượng mất đi *do streaming*, vì giới hạn của bản thân model là
như nhau ở mọi cấu hình. Mẫu: 128 s TTS tiếng Việt, `large-v3-turbo`, 12 thread, 469 từ tham chiếu.

| Cấu hình | WER | xoá / thay / thêm | số từ | wall |
|---|---|---|---|---|
| baseline (greedy final) | 0,431 | 182 / 20 / 0 | 293 | 53,8 s |
| `--temperature-inc 0.2` | 0,431 | 182 / 20 / 0 | 293 | 55,0 s |
| `--condition-on-previous` | 0,431 | 182 / 20 / 0 | 293 | 62,7 s |
| **`--beam-size 5`** | **0,250** | 84 / 21 / 12 | 403 | 58,4 s |
| beam 5 + temp fallback | 0,250 | — | — | 72,4 s |

Bốn điều rút ra:

1. **Greedy decode làm mất hẳn nội dung ở utterance dài** — 182 lần xoá trên 469 từ nhưng
   **0 lần thêm**: đó là dấu hiệu bỏ nguyên đoạn, không phải nghe sai. `beam_size = 5` hạ WER
   42% tương đối và lấy lại 110 từ. Đã thành **mặc định** cho lượt Final; partial vẫn greedy.
   Giá phải trả đo được: Final 9 573 ms → 10 497 ms (+9,6%), partial **không đổi** (1 883 vs
   1 873 ms) vì beam chỉ áp cho Final.
2. **Không phải lỗi của VAD.** Chạy lại với `--vad-gate 0` và với energy VAD thay Silero cho ra
   số **giống hệt** (0,431 / 293 từ / 5 utterance) — loại trừ hẳn nhánh VAD trước khi sửa decode.
3. **Temperature fallback vô tác dụng ở đây**: nó chỉ kích hoạt khi entropy/logprob vượt ngưỡng,
   mà kết quả greedy vẫn "qua" ngưỡng dù đã bỏ nội dung. Giữ mặc định tắt.
4. **Mồi prompt bằng text trước đó cũng không đổi độ chính xác** nhưng đắt thêm 17% thời gian.
   Giữ mặc định tắt (`session.condition_on_previous`).

Ổn định text sống với LocalAgreement: phần `stable_text` chỉ dài ra chứ không bị viết lại
(bất biến của `insert`, có test; `slide()` giữ phần đã chốt khi trượt cửa sổ). Đo trên full file:
105/111 lần cập nhật trong cùng một utterance giữ nguyên tiền tố, 6 ngoại lệ đều nằm ở ranh giới
utterance — nơi phần chốt reset một cách hợp lệ. Trung bình mỗi lần cập nhật: 14 từ đã chốt +
17 từ còn có thể đổi.

Hạn chế của phép đo cần nói rõ: oracle cũng dùng beam 5, nên WER-so-với-oracle có lợi cho cấu
hình beam. Tôi thử kiểm chứng bằng ground truth thật nhưng text chính xác duy nhất lấy được
(phần sapo) dài hơn hẳn đoạn audio 7,4 s tương ứng, nên không dùng làm reference được — con số
tuyệt đối vì thế còn để ngỏ. Cơ chế thì đã được xác nhận độc lập: 182 lần xoá / 0 lần thêm, và
trên đoạn ngắn 7,4 s greedy với beam cho kết quả y hệt nhau, đúng như dự đoán "chỉ utterance dài
mới bị".

## Kỹ thuật tham chiếu từ RealtimeSTT

[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) là một engine streaming STT bằng
Python đã chạy thực tế; bốn kỹ thuật của nó đã được cài lại (bằng Rust, theo kiến trúc
ở đây) sau khi đối chiếu với số đo của chính dự án này:

| Kỹ thuật | Ở RealtimeSTT | Ở đây |
|---|---|---|
| Model riêng cho partial | `realtime_model_type`, `beam_size_realtime` | `[partial_model]` / `--partial-model`, `SessionEngines` |
| VAD hai tầng | WebRTC làm cổng, Silero xác nhận | `GatedProbe`: cổng năng lượng + Silero, `vad.energy_gate_threshold` |
| Pre-roll dài | `pre_recording_buffer_duration = 1.0` | `session.pre_roll_secs` nâng 0,4 → 1,0 |
| Chặn độ trễ tích luỹ | `allowed_latency_limit` (100 chunk) | `session.max_probe_backlog_secs = 2.0` |

Vì sao **model riêng cho partial** là thứ đáng làm nhất: theo bảng đo ở trên, turbo cần
~2 s cho một cửa sổ partial 6 s, tức RTF ~1 chỉ riêng cho partial — chạy full 128 s audio
ở nhịp thời gian thực bị tụt lại 26 s. Ghép `large-v3-turbo` (final) với `base` (partial)
giữ nguyên chất lượng câu chốt mà partial về mức ~250 ms.

### Đo dual-model trên full file 128 s (nhịp thời gian thực)

| Cấu hình | wall | partial | median RTF partial | final | RSS |
|---|---|---|---|---|---|
| turbo 12 thread (một model) | 150,9 s | 52 lượt | 0,31 (~1,9 s) | 5 | 2,02 GB |
| turbo 10 thread + base 4 thread | **147,2 s** | **127 lượt** | **0,05 (~0,3 s)** | 5 | 2,23 GB |
| turbo 12 + base 12, hạn mức riêng | 221,4 s | 29 lượt | 0,04 nhưng max 17,0 | 5 | 2,23 GB |

Dòng thứ ba là bản sai (mỗi scheduler một semaphore) — xem ngay dưới. Dòng thứ hai là
bản đúng: text sống mượt hơn **2,4×** (127 so với 52 lượt cập nhật), độ trễ partial
giảm **6×**, chất lượng câu chốt không đổi vì final vẫn là turbo, đổi lại 210 MB RAM
cho model thứ hai. Có model partial nhanh rồi thì hạ `session.partial_interval_ms`
xuống 250 (RealtimeSTT để 200 ms) để text cập nhật dày hơn nữa.

### Cái bẫy khi ghép hai model: hạn mức thread phải dùng chung

Lần đo đầu tiên với dual-model **chậm hơn** turbo-only (221 s so với 158 s cho 128 s
audio): mỗi scheduler có semaphore riêng, nên turbo 12 thread và base 12 thread chạy
đồng thời thành 24 thread trên 16 core — đúng cái sập oversubscription đã đo ở mục
trên. Vì thế `InferenceScheduler` giờ nhận một [`ThreadBudget`] dùng chung ở cấp tiến
trình: mỗi lượt inference xin đúng `n_threads` permit, nên bất biến
`tổng thread đang chạy ≤ cpu_thread_budget` được giữ kể cả khi có nhiều model.

Chia thread cho dual-model: muốn partial chạy **chồng** được với final thì tổng phải
nằm trong hạn mức — máy 16 core (hạn mức 14) thì final 10 thread + partial 4 thread.
Nếu tổng vượt hạn mức, partial phải chờ final xong: vẫn đúng, nhưng mất hết cái lợi.

Hai thứ **không** lấy: `early_transcription_on_silence` (chạy final sớm một cách suy đoán
rồi bỏ nếu người nói tiếp) — trên CPU vốn đã bão hoà thì nhân đôi công việc mỗi lượt nói
là lỗ; và wake word — ngoài phạm vi một server ASR.

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

#!/usr/bin/env bash
# Thực nghiệm độ chính xác: dựng tham chiếu "oracle" bằng một lượt decode offline
# chất lượng cao, rồi đo WER của từng cấu hình streaming so với nó.
#
# Đo cái gì: phần chất lượng MẤT ĐI do chạy streaming (cắt câu theo VAD, cửa sổ
# partial, greedy...) — tách khỏi giới hạn của bản thân model, vốn giống nhau ở mọi
# cấu hình. Có ground truth thật thì truyền REF=<file> để đo WER tuyệt đối.
#
#   ./scripts/experiment_accuracy.sh models/ggml-large-v3-turbo.bin samples/mau.mp3
set -euo pipefail

MODEL="${1:?cần model}"
AUDIO="${2:?cần file audio}"
BIN="${BIN:-./target/release/whisper-rt}"
VAD="${VAD:-models/ggml-silero-v5.1.2.bin}"
LANG="${LANG_CODE:-vi}"
OUT_DIR="${OUT_DIR:-experiments}"
THREADS="${THREADS:-12}"

mkdir -p "$OUT_DIR"
ORACLE="$OUT_DIR/oracle.txt"

if [[ ! -s "$ORACLE" ]]; then
  echo "== dựng oracle (offline, beam 5, temperature fallback) ==" >&2
  "$BIN" --offline --file "$AUDIO" --model "$MODEL" --language "$LANG" \
    --threads "$THREADS" --beam-size 5 --temperature-inc 0.2 >"$ORACLE" 2>"$OUT_DIR/oracle.stderr"
fi
REF="${REF:-$ORACLE}"

run_case() {
  local name="$1"; shift
  local log="$OUT_DIR/$name.log"
  echo "== $name ==" >&2
  /usr/bin/time -f '%e' -o "$OUT_DIR/$name.wall" \
    "$BIN" --file "$AUDIO" --model "$MODEL" --vad-model "$VAD" --language "$LANG" \
    --threads "$THREADS" --no-realtime --wer "$REF" "$@" >"$log" 2>/dev/null || true
  local wer wall
  wer=$(grep -o 'WER=[0-9.]*' "$log" | tail -1 | cut -d= -f2)
  wall=$(cat "$OUT_DIR/$name.wall" 2>/dev/null)
  printf '%-28s WER=%-8s wall=%ss\n' "$name" "${wer:-?}" "${wall:-?}"
}

echo
run_case baseline
run_case temp-fallback     --temperature-inc 0.2
run_case beam5             --beam-size 5
run_case conditioned       --condition-on-previous
run_case beam5-temp        --beam-size 5 --temperature-inc 0.2
run_case beam5-temp-cond   --beam-size 5 --temperature-inc 0.2 --condition-on-previous

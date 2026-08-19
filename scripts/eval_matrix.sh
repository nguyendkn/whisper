#!/usr/bin/env bash
# Đo WER thật trên bộ eval có ground truth (manifest TSV), quét các cấu hình decode
# cho từng thứ tiếng. In một dòng `RESULT <lang> <case> WER=... rtf=...` cho mỗi ô.
#
#   ./scripts/eval_matrix.sh models/ggml-large-v3-turbo.bin eval/en.tsv en
set -euo pipefail

MODEL="${1:?cần model}"
MANIFEST="${2:?cần manifest TSV}"
LANG_CODE="${3:?cần mã ngôn ngữ}"
BIN="${BIN:-./target/release/whisper-rt}"
THREADS="${THREADS:-12}"

# LOG_DIR: nơi lưu kết quả từng clip (để tính khoảng tin cậy bootstrap sau).
LOG_DIR="${LOG_DIR:-}"

run() {
  local name="$1"; shift
  local tag="$LANG_CODE-$(basename "$MODEL" .bin)-$name"
  local out="${LOG_DIR:+$LOG_DIR/$tag.log}"
  local raw
  raw=$("$BIN" --eval-manifest "$MANIFEST" --model "$MODEL" --language "$LANG_CODE" \
    --threads "$THREADS" --eval-verbose "$@" 2>/dev/null)
  [[ -n "$out" ]] && printf '%s\n' "$raw" >"$out"
  echo "RESULT $LANG_CODE $(basename "$MODEL" .bin) $name $(printf '%s\n' "$raw" | grep '^EVAL' || echo 'EVAL failed')"
}

# CASES cho phép chọn tập cấu hình: "beam5 greedy temp-fallback beam8"
for case in ${CASES:-beam5 greedy temp-fallback}; do
  case "$case" in
    beam5)         run beam5 ;;
    greedy)        run greedy        --beam-size 1 ;;
    beam8)         run beam8         --beam-size 8 ;;
    temp-fallback) run temp-fallback --temperature-inc 0.2 ;;
    *) echo "bỏ qua case lạ: $case" >&2 ;;
  esac
done

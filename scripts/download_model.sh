#!/usr/bin/env bash
# Tải model whisper (GGML) và model Silero VAD của whisper.cpp về ./models.
#
#   ./scripts/download_model.sh                  # large-v3-turbo + VAD
#   ./scripts/download_model.sh base.en          # model khác
#   MODELS_DIR=/data/models ./scripts/download_model.sh
set -euo pipefail

MODEL="${1:-large-v3-turbo}"
MODELS_DIR="${MODELS_DIR:-models}"
BASE_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
# Model VAD nằm ở repo riêng, không cùng repo với model ASR.
VAD_BASE_URL="https://huggingface.co/ggml-org/whisper-vad/resolve/main"
VAD_MODEL="ggml-silero-v5.1.2.bin"

mkdir -p "$MODELS_DIR"

fetch() {
  local name="$1" base="${2:-$BASE_URL}" dest="$MODELS_DIR/$1"
  if [[ -f "$dest" ]]; then
    echo "đã có $dest, bỏ qua"
    return
  fi
  echo "tải $name ..."
  curl -fL --progress-bar -o "$dest.part" "$base/$name"
  mv "$dest.part" "$dest"
}

fetch "ggml-${MODEL}.bin"
fetch "$VAD_MODEL" "$VAD_BASE_URL"

echo
echo "xong. Cập nhật config/default.toml:"
echo "  [model] path = \"$MODELS_DIR/ggml-${MODEL}.bin\""
echo "  [vad]   model_path = \"$MODELS_DIR/$VAD_MODEL\""

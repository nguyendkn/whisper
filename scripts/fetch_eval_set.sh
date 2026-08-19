#!/usr/bin/env bash
# Tải bộ eval có ground truth từ FLEURS (Google, CC-BY 4.0) qua HF datasets-server và
# ghi manifest TSV `đường_dẫn_tuyệt_đối<TAB>text` cho `whisper-rt --eval-manifest`.
#
#   ./scripts/fetch_eval_set.sh                    # en_us + vi_vn, 30 clip mỗi thứ tiếng
#   ./scripts/fetch_eval_set.sh "en_us:en vi_vn:vi ja_jp:ja" 50 /data/eval
#
# Đường dẫn trong manifest phải là tuyệt đối: binary thường chạy ở thư mục khác.
set -euo pipefail

PAIRS="${1:-en_us:en vi_vn:vi}"
COUNT="${2:-30}"
OUT_DIR="$(cd "$(dirname "${3:-eval}")" && pwd)/$(basename "${3:-eval}")"

mkdir -p "$OUT_DIR/audio"

for pair in $PAIRS; do
  config="${pair%%:*}"
  lang="${pair##*:}"
  echo "== FLEURS $config -> $OUT_DIR/$lang.tsv" >&2
  CONFIG="$config" LANG_TAG="$lang" COUNT="$COUNT" OUT_DIR="$OUT_DIR" python3 - <<'PY'
import json, os, subprocess

config, lang = os.environ["CONFIG"], os.environ["LANG_TAG"]
count, out_dir = int(os.environ["COUNT"]), os.environ["OUT_DIR"]
url = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset=google%2Ffleurs&config={config}&split=validation&offset=0&length={count}"
)
raw = subprocess.run(["curl", "-sS", "--max-time", "120", url], capture_output=True, text=True).stdout
rows = json.loads(raw).get("rows", [])

lines, audio_secs = [], 0
for entry in rows:
    row = entry["row"]
    dest = os.path.join(out_dir, "audio", f'{config}-{row["id"]}.wav')
    if not os.path.exists(dest) or os.path.getsize(dest) < 2_000:
        subprocess.run(["curl", "-sSL", "--max-time", "90", "-o", dest, row["audio"][0]["src"]], check=False)
    if not os.path.exists(dest) or os.path.getsize(dest) < 2_000:
        print(f"  bỏ {dest}: tải không được")
        continue
    # `transcription` đã chuẩn hoá sẵn (chữ thường, không dấu câu).
    lines.append(f'{dest}\t{row["transcription"]}')
    audio_secs += row["num_samples"] / 16_000

manifest = os.path.join(out_dir, f"{lang}.tsv")
with open(manifest, "w", encoding="utf8") as handle:
    handle.write("\n".join(lines) + "\n")
words = sum(len(line.split("\t")[1].split()) for line in lines)
print(f"  {len(lines)} clip, {audio_secs:.0f} s audio, {words} từ tham chiếu")
PY
done

echo >&2
echo "Chạy eval:" >&2
echo "  ./target/release/whisper-rt --eval-manifest $OUT_DIR/vi.tsv \\" >&2
echo "      --model models/ggml-large-v3-turbo.bin --language vi --threads 12" >&2

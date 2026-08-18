#!/usr/bin/env bash
# Vòng lặp tối ưu: quét tham số inference, in bảng kết quả sắp theo hiệu năng.
#
#   ./scripts/bench_sweep.sh models/ggml-tiny.bin samples/dantri-body.mp3
#
# Biến môi trường: THREADS_LIST, CONCURRENCY_LIST, REPEAT, UTTERANCE_SECS,
# PARTIAL_WINDOW, BIN (đường dẫn binary), OUT (file kết quả),
# PIN (danh sách core cho taskset, ví dụ "0-5" để ghim vào P-core),
# CPU_BUDGET (tổng số thread cho phase 2, mặc định nproc), SKIP_PHASE1=1.
set -euo pipefail

MODEL="${1:?cần đường dẫn model}"
AUDIO="${2:?cần file audio}"
BIN="${BIN:-./target/release/whisper-rt}"
OUT="${OUT:-bench-results.txt}"
THREADS_LIST="${THREADS_LIST:-1 2 4 8 16}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 8}"
REPEAT="${REPEAT:-3}"
UTTERANCE_SECS="${UTTERANCE_SECS:-20}"
PARTIAL_WINDOW="${PARTIAL_WINDOW:-6}"
CPU_BUDGET="${CPU_BUDGET:-$(nproc)}"
PIN="${PIN:-}"

# Ghim vào một nhóm core cụ thể — trên CPU hybrid (P-core + E-core) đây là cách
# duy nhất để giữ inference khỏi rơi xuống E-core và kéo cả graph chậm theo.
if [[ -n "$PIN" ]]; then
  RUNNER=(taskset -c "$PIN" "$BIN")
else
  RUNNER=("$BIN")
fi

: >"$OUT"

run() {
  local threads="$1" concurrency="$2" extra="${3:-}"
  echo "→ threads=$threads concurrency=$concurrency ${extra:-(audio_ctx scaling on)}" >&2
  "${RUNNER[@]}" --bench --file "$AUDIO" --model "$MODEL" \
    --threads "$threads" --concurrency "$concurrency" \
    --repeat "$REPEAT" --utterance-secs "$UTTERANCE_SECS" \
    --partial-window "$PARTIAL_WINDOW" $extra 2>/dev/null \
    | grep '^BENCH' | tee -a "$OUT"
}

if [[ "${SKIP_PHASE1:-0}" != "1" ]]; then
  echo "== Phase 1: độ trễ theo số thread (1 stream) ==" >&2
  for threads in $THREADS_LIST; do
    run "$threads" 1
    run "$threads" 1 --no-audio-ctx-scaling
  done
fi

# Số thread cho partial nhanh nhất -> dùng cho phase 2.
BEST_THREADS=$(python3 - "$OUT" <<'PY'
import re, sys
best, best_ms = None, None
for line in open(sys.argv[1]):
    if 'kind=partial' not in line or 'audio_ctx_scaling=true' not in line:
        continue
    f = dict(p.split('=', 1) for p in line.split() if '=' in p)
    ms = int(f['median_ms'])
    if best_ms is None or ms < best_ms:
        best, best_ms = f['threads'], ms
print(best or 4)
PY
)
# Phase 2 giữ TỔNG số thread không đổi: streams * threads <= CPU_BUDGET.
# Tăng streams mà giữ nguyên threads là cách chắc chắn nhất để mọi session cùng
# chậm — ggml spin-wait ở barrier nên oversubscribe không xuống dốc từ từ mà sập.
echo "== Phase 2: throughput, tổng thread cố định = $CPU_BUDGET ==" >&2
for concurrency in $CONCURRENCY_LIST; do
  threads=$(( CPU_BUDGET / concurrency ))
  [[ "$threads" -lt 1 ]] && threads=1
  run "$threads" "$concurrency"
done

echo >&2
python3 - "$OUT" <<'PY'
import sys
rows = []
for line in open(sys.argv[1]):
    if not line.startswith('BENCH'):
        continue
    rows.append(dict(p.split('=', 1) for p in line.split() if '=' in p))

def table(kind, cols, sort_key):
    sel = [r for r in rows if r.get('kind') == kind]
    if not sel:
        return
    sel.sort(key=sort_key)
    print(f'\n{kind.upper()}')
    print('  ' + '  '.join(f'{c:>18}' for c in cols))
    for r in sel:
        print('  ' + '  '.join(f'{r.get(c, "-"):>18}' for c in cols))

table('partial', ['threads', 'audio_ctx_scaling', 'median_ms', 'worst_ms', 'rtf'],
      lambda r: int(r['median_ms']))
table('final', ['threads', 'audio_ctx_scaling', 'median_ms', 'worst_ms', 'rtf'],
      lambda r: int(r['median_ms']))
table('throughput', ['threads', 'streams', 'wall_ms', 'aggregate_rtf', 'streams_at_rtf1'],
      lambda r: -float(r['streams_at_rtf1']))
PY

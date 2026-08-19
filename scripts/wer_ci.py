#!/usr/bin/env python3
"""Tính WER toàn corpus kèm khoảng tin cậy bootstrap từ log eval từng clip.

Đầu vào là các file log do `eval_matrix.sh` sinh ra (LOG_DIR), mỗi dòng clip có dạng
`<tên> WER=... sub=N del=N ins=N ref=N`.

Bootstrap lấy mẫu lại **theo clip** (không theo từ): clip là đơn vị độc lập, còn các
từ trong cùng một clip thì tương quan với nhau. So sánh hai cấu hình thì dùng bootstrap
theo cặp trên đúng tập clip chung — cách duy nhất để nói chênh lệch có thật hay là nhiễu.

    ./scripts/wer_ci.py eval-logs/*.log
    ./scripts/wer_ci.py --compare eval-logs/vi-a.log eval-logs/vi-b.log
"""
import random
import re
import sys

CLIP = re.compile(r"^(\S+)\s+WER=[\d.]+\s+sub=(\d+)\s+del=(\d+)\s+ins=(\d+)\s+ref=(\d+)")


def load(path):
    """{tên clip: (số lỗi, số từ tham chiếu)}"""
    clips = {}
    with open(path, encoding="utf8") as handle:
        for line in handle:
            m = CLIP.match(line.strip())
            if m:
                name, sub, dele, ins, ref = m.group(1), *map(int, m.groups()[1:])
                clips[name] = (sub + dele + ins, ref)
    return clips


def corpus_wer(clips, keys=None):
    keys = keys if keys is not None else list(clips)
    errors = sum(clips[k][0] for k in keys)
    words = sum(clips[k][1] for k in keys)
    return errors / words if words else 0.0


def bootstrap(clips, rounds=5000, seed=1):
    rng = random.Random(seed)
    keys = list(clips)
    samples = []
    for _ in range(rounds):
        picked = [rng.choice(keys) for _ in keys]
        samples.append(corpus_wer(clips, picked))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    return lo, hi


def compare(path_a, path_b, rounds=5000, seed=1):
    a, b = load(path_a), load(path_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        print("không có clip chung")
        return
    rng = random.Random(seed)
    diff = corpus_wer(b, shared) - corpus_wer(a, shared)
    wins = 0
    samples = []
    for _ in range(rounds):
        picked = [rng.choice(shared) for _ in shared]
        delta = corpus_wer(b, picked) - corpus_wer(a, picked)
        samples.append(delta)
        wins += delta > 0
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    print(f"clip chung: {len(shared)}")
    print(f"  A {path_a}: WER={corpus_wer(a, shared):.4f}")
    print(f"  B {path_b}: WER={corpus_wer(b, shared):.4f}")
    print(f"  B - A = {diff:+.4f}  KTC 95% [{lo:+.4f}, {hi:+.4f}]")
    p = min(wins, rounds - wins) / rounds * 2
    verdict = "CÓ THẬT" if lo > 0 or hi < 0 else "KHÔNG kết luận được (nhiễu)"
    print(f"  p≈{p:.3f} -> {verdict}")


def main():
    args = sys.argv[1:]
    if args[:1] == ["--compare"]:
        compare(args[1], args[2])
        return
    for path in args:
        clips = load(path)
        if not clips:
            continue
        wer = corpus_wer(clips)
        lo, hi = bootstrap(clips)
        words = sum(c[1] for c in clips.values())
        print(f"{path}: clip={len(clips)} từ={words} WER={wer:.4f} KTC 95% [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()

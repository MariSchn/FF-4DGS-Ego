"""Paired, sequence-clustered comparison of two held-out evaluations.

96 target frames are not 96 independent samples: frames share clips and clips share sequences.
The unit here is the clip, deltas are paired on (seq, offset), and the bootstrap resamples
SEQUENCES with replacement, so correlated clips cannot manufacture confidence.

    python -m scripts.paired_clip_stats results/heldout_on.json results/heldout_init48.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def paired(a: dict, b: dict, key: str):
    ra = {(r["seq"], r["offset"]): r.get(key) for r in a["per_clip"]}
    rb = {(r["seq"], r["offset"]): r.get(key) for r in b["per_clip"]}
    common = [k for k in ra if k in rb and ra[k] is not None and rb[k] is not None]
    return {k: ra[k] - rb[k] for k in common}


def cluster_bootstrap(deltas: dict, iters: int = 10000, seed: int = 0):
    by_seq = defaultdict(list)
    for (seq, _), d in deltas.items():
        by_seq[seq].append(d)
    seqs = sorted(by_seq)
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        pick = [rng.choice(seqs) for _ in seqs]
        pooled = [d for s in pick for d in by_seq[s]]
        means.append(sum(pooled) / len(pooled))
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="json of the arm under test")
    ap.add_argument("b", help="json of the reference")
    ap.add_argument("--keys", nargs="+", default=["psnr", "hand_psnr", "bg_psnr", "lpips"])
    a = ap.parse_args()
    A, B = json.load(open(a.a)), json.load(open(a.b))
    print(f"{A['label']}  vs  {B['label']}")
    for key in a.keys:
        d = paired(A, B, key)
        if not d:
            print(f"  {key:10s}: no paired clips")
            continue
        vals = sorted(d.values())
        n = len(vals)
        mean = sum(vals) / n
        med = vals[n // 2]
        wins = sum(v > 0 for v in vals)
        lo, hi = cluster_bootstrap(d)
        n_seq = len({s for (s, _) in d})
        sig = "significant" if (lo > 0 or hi < 0) else "NOT significant"
        print(f"  {key:10s}: n={n} clips / {n_seq} seqs  mean {mean:+.3f}  median {med:+.3f}  "
              f"wins {wins}/{n}  95% CI [{lo:+.3f}, {hi:+.3f}]  {sig}")


if __name__ == "__main__":
    main()

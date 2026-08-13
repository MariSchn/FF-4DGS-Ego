#!/usr/bin/env python3
"""Cross-check our HOI4D split against the one the Hand3R authors sent us, and emit the clip set
that is genuinely held out for BOTH methods.

WHY THIS EXISTS. Our paper declines to tabulate against Hand3R's published 42.6 mm, and gives four
reasons: joint count, segment length, an unpublished split, and HOI4D fine-tuning. On 2026-08-13 the
Hand3R authors sent us the split, which turns the third reason from a caveat into a measurement.
The measurement is worse than the caveat was:

  * Hand3R fine-tunes on 1,271 of the 1,682 HOI4D clip directories it uses, i.e. 75.6%. Any HOI4D
    test set that is not their exact held-out split is therefore, in expectation, three-quarters
    their training data. Ours is 67.5%.
  * The reverse holds too. A third of their 300-clip paper subset sits in OUR training set, so we
    cannot simply adopt their split either.
  * Their split unit is the scene (subject, category, scene) precisely so that different takes of
    one scene cannot straddle the boundary. Ours is the take. Under their unit, 154 of our 157 test
    sequences share a scene with our own training set.

The first two make a direct comparison meaningless in a way no footnote repairs, and they point at
the fix: score both methods on clips that are held out under BOTH disciplines at once. That set is
what this script computes, as ``--out``.

    python -m scripts.hand3r_protocol.audit_split_overlap \
        --hand3r_split_dir ~/Downloads/Hand3R_evaluation_protocol/splits \
        --our_test  <(ls $S/hoi4d_test157_detv3) \
        --our_train <(ls $S/hoi4d_train) \
        --out report/hand3r_fair_eval_clips.txt
"""
from __future__ import annotations

import argparse
import json
import os


def scene_of(clip: str) -> tuple[str, str, str]:
    """``ZY20210800001_H1_C12_N26_S165_s01_T1`` -> ``(H1, C12, S165)``.

    This is Hand3R's split unit, verbatim from their PROTOCOL.md: "(H, C, S) = (subject, category,
    scene). All clips/takes from one such group stay in the same split." Note it deliberately omits
    N (the object instance) and the take suffix, so it is COARSER than the (C, N, S, s) tuple our
    own 2026-07-03 leakage audit used. That audit concluded Hand3R's split "has the same property"
    as ours; the file they sent refutes it, and the difference is 25 sequences against 154.
    """
    parts = clip.split("_")
    return parts[1], parts[2], parts[4]


def read_lines(path: str) -> set[str]:
    with open(path) as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand3r_split_dir", required=True)
    ap.add_argument("--our_test", required=True, help="one clip id per line")
    ap.add_argument("--our_train", required=True)
    ap.add_argument("--out", default=None, help="write the mutually-held-out clip list here")
    ap.add_argument("--from_paper300", action="store_true",
                    help="restrict the fair set to their 300-clip paper subset, so their published "
                         "number is reproducible on it, instead of their full 411-clip held-out set")
    a = ap.parse_args()

    info = json.load(open(os.path.join(a.hand3r_split_dir, "split_info.json")))
    h3_train, h3_val = set(info["train_dirs"]), set(info["val_dirs"])
    h3_300 = read_lines(os.path.join(a.hand3r_split_dir, "paper_eval_clips.txt"))
    our_test, our_train = read_lines(a.our_test), read_lines(a.our_train)
    our_train_scenes = {scene_of(c) for c in our_train}

    n = len(our_test)
    print(f"Hand3R fine-tunes on {len(h3_train)}/{len(h3_train) + len(h3_val)} clip dirs "
          f"= {100 * len(h3_train) / (len(h3_train) + len(h3_val)):.1f}% of the HOI4D it uses\n")

    print("A. our test set, seen through their split")
    for label, hit in (("in their TRAINING set", our_test & h3_train),
                       ("in their held-out set", our_test & h3_val),
                       ("in neither", our_test - h3_train - h3_val)):
        print(f"     {label:24s} {len(hit):4d}  ({100 * len(hit) / max(n, 1):5.1f}%)")

    print("\nB. their paper subset, seen through ours")
    print(f"     of their 300, we TRAINED ON  {len(h3_300 & our_train):4d}  "
          f"({100 * len(h3_300 & our_train) / max(len(h3_300), 1):5.1f}%)")

    print("\nC. our own split, judged by their scene rule")
    sib = {c for c in our_test if scene_of(c) in our_train_scenes}
    print(f"     our test clips sharing a SCENE with our train  {len(sib):4d}  "
          f"({100 * len(sib) / max(n, 1):5.1f}%)")
    print(f"     scene-disjoint under their rule                {n - len(sib):4d}")

    pool = h3_300 if a.from_paper300 else h3_val
    fair = sorted(c for c in pool - our_train if scene_of(c) not in our_train_scenes)
    print(f"\nD. held out for BOTH methods "
          f"({'their paper-300' if a.from_paper300 else 'their full 411'} - our train, scene-disjoint)")
    print(f"     {len(fair)} clips over {len({scene_of(c) for c in fair})} scenes")
    print(f"     already converted in our store: {len(set(fair) & our_test)}")

    if a.out:
        with open(a.out, "w") as fh:
            fh.write("\n".join(fair) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

"""Every consumer of `clip_cams` must survive its tuple gaining a field.

WHAT THIS CAUGHT (2026-08-07). `predict_clip` returns a 5-tuple
`(pj_cam, c2w, s_clip, ratios, s_failed)`; `s_failed` was added for task #63. Two consumers in
`eval_sequence` were never updated and still unpacked FOUR:

    for c, (pj, _, _, _) in enumerate(clip_cams):                       # camera-frame C-MPJPE
    for (pj, c2w, _, _), c2wg in zip(clip_cams, clip_grav)              # gravity oracle

The first killed **every** sequence of geo59's seg100 stage with
`ValueError: too many values to unpack (expected 4)`, exit 1, after the seg30 stage had already
completed fine on the older file. It stayed latent because the process that ran seg30 had imported
the pre-change module, and only the freshly-started seg100 stage picked up the newer one.

A grep is not enough here: the failure is a plain arity mismatch that no type checker in this repo
would catch, and the paths are guarded by flags (`gravity_oracle`) so they are not exercised by a
normal run. This test asserts the property at source level instead: no consumer may hard-code the
arity.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "scripts" / "eval_world_space.py").read_text()


def _predict_clip_arity() -> int:
    """How many values predict_clip actually returns."""
    m = re.search(r"\n    return (pred_joints\[0\][^\n]*)\n", SRC)
    assert m, "could not find predict_clip's return statement"
    # top-level comma split is fine here: the return is a flat tuple of simple expressions
    return len([p for p in m.group(1).split(",") if p.strip()])


def test_predict_clip_returns_five_values():
    """Pins the contract the consumers below depend on."""
    assert _predict_clip_arity() == 5, (
        f"predict_clip returns {_predict_clip_arity()} values; if this changed deliberately, every "
        "clip_cams consumer must be re-checked and this test updated in the same commit")


def test_no_consumer_hardcodes_a_shorter_arity():
    """Any `(a, b, c, d)`-style destructure of clip_cams with the wrong count is a latent crash."""
    bad = []
    for line_no, line in enumerate(SRC.splitlines(), start=1):
        if "clip_cams" not in line:
            continue
        # Find tuple DESTRUCTURES like `(pj, _, _, _)`. A parenthesis preceded by an
        # identifier character is a CALL argument list (`zip(...)`, `_world_from_cam(...)`), not a
        # destructure, and must not be flagged - that false positive is what made the first
        # version of this test useless.
        for m in re.finditer(r"\(([^()]*)\)", line):
            if m.start() > 0 and re.match(r"[A-Za-z0-9_]", line[m.start() - 1]):
                continue                              # call, not a destructure
            pat = m.group(1)
            parts = [p.strip() for p in pat.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            if any(p.startswith("*") for p in parts):
                continue                              # starred: arity-proof, fine
            if not any(p == "_" for p in parts):
                # HEURISTIC, and a deliberate one. A line may destructure more than one thing
                # (`for (dp, dv), (_, c2w, ...) in zip(clip_dense, clip_cams)`), and only the
                # clip_cams half is in scope here. Positional clip_cams unpacks always discard
                # fields with bare `_`; the companion tuples (`(dp, dv)`) do not. Requiring at
                # least one bare `_` selects the clip_cams destructure without needing to resolve
                # which side of a zip() it came from.
                continue
            if all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in parts):
                if len(parts) != _predict_clip_arity():
                    bad.append(f"  line {line_no}: ({pat}) unpacks {len(parts)}, "
                               f"predict_clip returns {_predict_clip_arity()}\n    {line.strip()}")
    assert not bad, (
        "clip_cams destructured with the wrong arity - this raises "
        "'ValueError: too many values to unpack' the moment the branch runs:\n" + "\n".join(bad)
        + "\nUse starred unpacking, e.g. `(pj, *_rest)`, so a future field cannot break it.")

"""Is the shared-token frozen head the same as a whole frozen model?

The teacher runs the student's own backbone tokens through an untouched copy of the Gaussian head
with the injection hook off. That is equivalent to forwarding a separate frozen model only while
the trunk is frozen and deterministic. This checks the claim on a real batch rather than assuming
it, because if the two disagree the target is not what we think it is and the whole term is
misdirected.

Run before enabling gs_depth_target: frozen_teacher on anything that matters.
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml

from scripts.eval_hand_head import build_model, load_hand_head
from scripts.run_ours_gs import load_clip
from scripts.train_hand_head import build_views

BOX_FILE = "hand_bboxes_v2_rf1.5_res224x224.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="student checkpoint; omit for the base itself")
    ap.add_argument("--export_root", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    student = build_model(cfg, a.device)
    if a.ckpt:
        load_hand_head(student, a.ckpt, a.device)
    student.eval()

    # attach the teacher exactly as the trainer does
    import copy
    teacher = copy.deepcopy(student.gs_head).to(a.device).eval()
    base = torch.load(cfg["model"]["checkpoint"], map_location=a.device)
    base = base.get("state_dict", base.get("reconstructor", base))
    pre = {k[len("gs_head."):]: v for k, v in base.items() if k.startswith("gs_head.")}
    if not pre:
        raise SystemExit("base checkpoint holds no gs_head weights")
    teacher.load_state_dict(pre, strict=False)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student._teacher_gs_head = [teacher]

    # the oracle: a whole separate model straight from the base, with the injection disabled
    ref_cfg = yaml.safe_load(open(a.config))
    ref_cfg["model"].setdefault("hand_to_gs_injection", {})["enabled"] = False
    reference = build_model(ref_cfg, a.device)
    reference.eval()

    imgs, frames = load_clip(os.path.join(a.export_root, a.seq, "images"), a.n_views)
    imgs = imgs.unsqueeze(0).to(a.device)
    n = imgs.shape[1]
    bb = torch.load(os.path.join(a.store, a.seq, "hand_data", BOX_FILE), map_location="cpu")
    fi = torch.tensor(frames, dtype=torch.long)
    hb = bb["bboxes"][fi].unsqueeze(0).to(a.device)
    hv = bb["valid"][fi].bool().unsqueeze(0).to(a.device)
    views = build_views(imgs, n, a.device, hb, hv, frame_index=fi.unsqueeze(0))

    with torch.no_grad():
        p_student = student(views, is_inference=False, use_motion=False)
        p_ref = reference(views, is_inference=False, use_motion=False)

    if "gs_depth_teacher_logit" not in p_student:
        raise SystemExit("no gs_depth_teacher_logit in preds: the teacher did not run")
    if "gs_depth_logit" not in p_ref:
        raise SystemExit("no gs_depth_logit in the reference preds")

    t = p_student["gs_depth_teacher_logit"].float()
    r = p_ref["gs_depth_logit"].float()
    if t.shape != r.shape:
        raise SystemExit(f"shape mismatch: teacher {tuple(t.shape)} vs reference {tuple(r.shape)}")
    d = (t - r).abs()
    print(f"teacher  median {float(t.median()):+.6f}  min {float(t.min()):+.4f}  max {float(t.max()):+.4f}")
    print(f"oracle   median {float(r.median()):+.6f}  min {float(r.min()):+.4f}  max {float(r.max()):+.4f}")
    print(f"abs diff max {float(d.max()):.3e}  mean {float(d.mean()):.3e}  "
          f"frac>1e-3 {float((d > 1e-3).float().mean()):.5f}")
    # bf16 round-trips through the trunk, so exact equality is not the bar; a systematic gap is.
    ok = float(d.max()) < 1e-2 and float(d.mean()) < 1e-4
    print("EQUIVALENT" if ok else "NOT EQUIVALENT: the shared-token teacher is not the frozen model")
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

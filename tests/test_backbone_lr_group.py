"""Unfrozen backbone blocks must be able to take their own learning rate.

WHAT THIS CAUGHT (2026-08-07). Job 9954605_1 unfroze the last VGGT block and trained it at the
head's `lr: 1e-4`, because `trainable_params` is one flat list and the optimizer took a single lr.
The arm looked decisively harmful:

    HOI4D  C_abs 72.46 -> 79.60   C_rr 36.48 -> 48.11

and it would have been reported as "freezing is better", which is the paper's own design point and
therefore exactly the claim we are least entitled to accept on weak evidence. But 1e-4 is a rate
tuned for a randomly-initialised head; fine-tuning a pretrained ViT is normally 1e-5 or lower,
often with layer-wise decay. Training a pretrained backbone at the head's lr is the standard way to
destroy it, so the run measured the learning rate, not the freezing.

Two properties are asserted:
  1. `training.backbone_lr` puts the unfrozen backbone in its own param group;
  2. when it is NOT set, the code says so loudly, because silence there is what produced a
     confounded P0 result that we nearly reported.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "scripts" / "train_hand_head.py").read_text()


def test_unfrozen_backbone_params_are_recorded():
    """The optimizer cannot group what the unfreeze step never handed it."""
    assert "model._unfrozen_backbone_params = backbone_unfrozen" in SRC, (
        "the unfreeze step must record its params so the optimizer can give them their own lr")


def test_backbone_lr_creates_a_separate_param_group():
    assert 'training_cfg.get("backbone_lr", base_lr)' in SRC, (
        "training.backbone_lr must be readable from the config")
    m = re.search(r"if backbone_params and backbone_lr != base_lr:(.{0,400}?)\n\s*if root_anchor",
                  SRC, re.S)
    assert m, "expected a backbone-specific param group guarded on backbone_lr != base_lr"
    assert '"lr": backbone_lr' in m.group(1), (
        "the backbone group must actually carry backbone_lr, not base_lr")


def test_param_groups_do_not_double_count_a_parameter():
    """A parameter in two groups gets stepped twice. The de-dup is what prevents that."""
    assert "seen" in SRC and "id(p) not in seen" in SRC, (
        "backbone and root_anchor groups must be de-duplicated against each other and against "
        "the base group, or a shared tensor would receive two updates per step")


def test_unfrozen_backbone_at_head_lr_warns_loudly():
    """The silent case is the one that cost us a confounded result, so it must not stay silent."""
    m = re.search(r"else:\n\s*optimizer = Adam\(trainable_params, lr=base_lr\)(.{0,600}?)\n\s*scheduler",
                  SRC, re.S)
    assert m, "could not find the single-group optimizer branch"
    body = m.group(1)
    assert "backbone_params" in body and "!!" in body, (
        "training an unfrozen backbone at the head's lr must print a loud warning")
    assert "backbone_lr" in body, "the warning must name the setting that fixes it"
    assert "not the freezing" in body or "not freezing" in body, (
        "the warning must say what the confound actually invalidates: a frozen-vs-unfrozen "
        "comparison run this way measures the learning rate")

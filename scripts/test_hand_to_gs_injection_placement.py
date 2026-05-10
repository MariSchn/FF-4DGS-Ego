"""Synthetic placement test for HandToGSInjection (per-layer variant).

Builds a 4-layer feats list of all-zero tensors at the canonical DPT post-projects
shapes, runs the injection with one valid + one invalid hand at a known bbox,
and verifies that for every layer:

    - feats_out[i] - feats[i] has non-zero entries strictly inside the bbox
      region of layer i (denormalized to that layer's resolution).
    - The diff is exactly zero outside that bbox.
    - The invalid hand contributes nothing (its bbox is the degenerate (0,0,0,0)).

Run:
    python -m scripts.test_hand_to_gs_injection_placement
"""

import torch

from diffsynth.auxiliary_models.worldmirror.models.heads.hand_to_gs_injection import (
    HandToGSInjection,
)


def main() -> int:
    torch.manual_seed(0)

    B, S = 1, 2
    N = B * S
    crop_size = 8
    hand_dim = 1024
    gs_dims = [256, 512, 1024, 1024]
    # Default ph=pw=37 (518/14), but the resize layers (k=4 stride 4 / k=2 s=2 /
    # identity / k=3 s=2) yield h_i = 4*ph, 2*ph, ph, ceil(ph/2) ≈ 19.
    ph = pw = 37
    layer_shapes = [(4 * ph, 4 * pw), (2 * ph, 2 * pw), (ph, pw), ((ph + 1) // 2, (pw + 1) // 2)]

    inj = HandToGSInjection(hand_dim=hand_dim, gs_dims=gs_dims, use_hand_valid_mask=True)
    inj.eval()

    enhanced_crop_tokens = torch.randn(N * 2, crop_size * crop_size, hand_dim)

    # Hand 0 valid (centered bbox), hand 1 invalid (degenerate bbox).
    bboxes = torch.zeros(B, S, 2, 4)
    bboxes[..., 0, :] = torch.tensor([0.25, 0.25, 0.75, 0.75])
    valid = torch.zeros(B, S, 2, dtype=torch.bool)
    valid[..., 0] = True

    feats = [
        torch.zeros(N, gs_dims[i], h_i, w_i) for i, (h_i, w_i) in enumerate(layer_shapes)
    ]

    with torch.no_grad():
        feats_out = inj(enhanced_crop_tokens, bboxes, valid, feats)

    all_pass = True
    for i, (feat_in, feat_out, (h_i, w_i)) in enumerate(zip(feats, feats_out, layer_shapes)):
        diff = feat_out - feat_in

        x1 = max(0, min(w_i, int(round(0.25 * w_i))))
        y1 = max(0, min(h_i, int(round(0.25 * h_i))))
        x2 = max(0, min(w_i, int(round(0.75 * w_i))))
        y2 = max(0, min(h_i, int(round(0.75 * h_i))))

        # Build a boolean mask of the bbox region across all (n, c, y, x).
        mask = torch.zeros_like(diff, dtype=torch.bool)
        for n in range(N):
            mask[n, :, y1:y2, x1:x2] = True

        inside = diff[mask]
        outside = diff[~mask]

        # Inside: non-zero (the projected crop scattered into this region).
        inside_nonzero = inside.abs().sum().item() > 0
        # Outside: exactly zero (no leakage).
        outside_zero = outside.abs().sum().item() == 0
        # Sanity: diff should be zero on the invalid hand's frame *if* and only if
        # we'd otherwise scatter — but since hand 1 has an all-zero bbox we just
        # confirm that no extra contribution leaked outside hand 0's bbox.

        layer_pass = inside_nonzero and outside_zero
        all_pass = all_pass and layer_pass

        print(
            f"layer {i}: shape=({gs_dims[i]:>4}, {h_i:>3}, {w_i:>3}) "
            f"bbox_pixels=[{x1},{y1},{x2},{y2}] "
            f"inside_nonzero={inside_nonzero} outside_zero={outside_zero} "
            f"-> {'PASS' if layer_pass else 'FAIL'}"
        )

    print("OVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

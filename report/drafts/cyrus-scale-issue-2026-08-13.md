# For Cyrus, spoken, about the scale in Sec. 3.3

---

You asked if the hand depth and the scene depth are compatible in metric scale. They are, and
that's the problem.

Our predicted camera trajectory is already metric, within 2% of the ground truth one. So `s` should
come out at 1. It comes out at 0.617.

That means the equation isn't resolving an unknown scale. It's putting a 38% error into a
trajectory that was already right, and every world number multiplies the camera translation by it.

Two sentences in that section say the hand resolves the scene's scale. Our numbers say otherwise,
and the ablation a reviewer will ask for is just s = 1.

I didn't rewrite it because it changes every world number and it's your call. Either we stop
applying `s` and report it as a consistency check between two independently metric estimates, or we
keep it and explain why a correction that makes things worse is right. I'd go with the first, and
it's arguably the better result anyway: it says the hand and the scene agree on scale without
either being told to.

---

Sources if he asks: `s_med` 0.617 and `s_pool` 0.550 in
`report/hoi4d_world_eval_2026-06-22_3scale.json`; the 1.02 from
`solve_similarity(pred_cam, gt_cam)` at `eval_world_space.py:652`; the code comment at `:500-505`
already calls the pooled hand scale biased and heavy-tailed.

Hi Chenyangguang, quick follow-up with actual numbers on the two scale routes we discussed.

I got both running and they both work, directionally.

(b) the feedforward scale head (your lighter suggestion): a small head predicts one global
metric scale per clip from the register token, trained so s*gs_depth matches the metric
hand. The scale residual drops monotonically as it trains (14.1 to 13.1 cm over the first
few steps), absolute MPJPE comes down with it, and the rendering and pose shape are
untouched (PSNR ~40 dB, PA-MPJPE ~8 mm). So a feedforward, no-per-clip-solve metric scale
is learnable.

(a) the partial unfreeze (last 4 blocks) with direct GT object-depth supervision: this is
the important one, because on a frozen backbone the object depth got worse (B2). With the
blocks unfrozen the object depth now trends down instead (23.8 to 22.9 cm over 6 steps),
hand depth and absolute MPJPE drop too, pose stays at PA ~5.5 mm. Opposite sign to the
frozen case, which is exactly what we wanted to see: unfreezing + a metric target pulls the
scene toward metric rather than distorting it. This matches the HOI4D dense-depth result
(scene depth 20.5 to 13.4 cm with no masking), which is the converged version of the same
effect.

Important caveat: both (a) and (b) numbers above are directional, from short runs on a few
sequences. The cluster storage quota is exhausted (can't write logs) so I'm streaming these
off node-local scratch, which caps each run at ~30 min, and (a) trains 100 M backbone
params so it moves slowly. The trends are clearly correct; the converged magnitudes (object
depth driven well below the frozen baseline, the scale-head floor) need a longer
uninterrupted run, which is where a bigger card would really help.

Net: the scale mechanism works and is learnable, the scene goes metric where we have a depth
target. I'm attaching the updated comparison table with these rows. Next I'll get the
HaWoR world-space baseline running for the comparison you asked for.

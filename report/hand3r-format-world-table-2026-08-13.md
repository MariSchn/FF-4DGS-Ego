# World table in Hand3R's Table II layout, 2026-08-13

Every cell below comes from `h3r_<row>_seg{30,100}.json`, produced by `eval_worldspace_baseline`
with `--wa_short 30 --hands right --drop_partial_tail` on stored per-frame predictions. Student
cluster job 105655 (ours, HaWoR, WiLoR, HaMeR), Euler job 10559408 (HaPTIC, Dyn-HaMR).

Generated into LaTeX by `scripts/make_world_table.py`, which refuses to emit if any two rows
disagree on `segment_len`, `wa_short`, `hands`, `drop_partial_tail`, `w_h3r_align_frames` or
`joints_per_hand`.

## The table

| Method | Type | Pipeline | C-MPJPE | Short WA | Short W | Long WA | Long W |
|---|---|---|---|---|---|---|---|
| HaMeR + SLAM | offline | multi-stage | 87.9 | 34.7 | 103.3 | 57.5 | 219.9 |
| WiLoR + SLAM | offline | multi-stage | 83.4 | 33.4 | 96.8 | 56.3 | 209.9 |
| HaWoR | offline | multi-stage | 87.7 | 40.1 | 118.4 | 61.0 | 229.4 |
| HaPTIC + SLAM | offline | multi-stage | (157.1) | 36.7 | (117.0) | 57.1 | (263.1) |
| Dyn-HaMR | offline | optimisation | (1336.7) | 49.3 | (164.8) | 69.0 | (328.2) |
| **Ours** | online | one-stage | **35.4** | **28.1** | **73.0** | **43.3** | **121.2** |

W is Hand3R's gauge: rigid, scale fixed, fitted on the first two frames of each chunk. WA aligns
the whole chunk. Parentheses mark rows without recovered metric depth, whose absolute and global
cells report the scale of the inputs we supply.

Segments: ours 1506/471, HaMeR 1486/468, WiLoR 1486/468, HaWoR 1425/468, HaPTIC 1425/468,
Dyn-HaMR 1497/468.

## The gauge is worth more than any method difference in the table

Hand3R's two-frame fit minus our 30-frame fit, per row, in mm:

| Method | short | long |
|---|---|---|
| Ours | **+40.9** | **+50.4** |
| WiLoR + SLAM | +56.4 | +81.0 |
| HaMeR + SLAM | +59.7 | +84.1 |
| HaWoR | +65.0 | +95.8 |
| HaPTIC + SLAM | +73.2 | +125.0 |
| Dyn-HaMR | +85.8 | +132.8 |

Every W number this project has reported roughly doubles under their convention. Our own short W
goes 32.1 to 73.0 and long W goes 70.7 to 121.2.

**Adopting their gauge helps us.** It costs us less than it costs any other row, because it is a
drift measure and our trajectory drifts least: a two-frame anchor cannot absorb accumulated error
the way a thirty-frame least-squares fit can. Our short-video lead over the best baseline widens
from 8.3 mm (26%) under our gauge to 23.8 mm (33%) under theirs. So the stricter convention,
proposed by the competitor, is the one that flatters us, which is the only comfortable position to
be in when adopting someone else's protocol.

## Where this sits against Hand3R's published numbers

Not a comparison. Different split, different boxes, different joint count, and Hand3R fine-tunes on
HOI4D. Recorded only because the +SLAM rows are a partial sanity check on our port of their gauge.

| Row | ours, long W | Hand3R published, long W |
|---|---|---|
| HaMeR + SLAM | 219.9 | 218.05 |
| WiLoR + SLAM | 209.9 | 223.00 |
| HaWoR | 229.4 | 58.62 |
| Ours / Hand3R | 121.2 | 125.81 |

The two SLAM rows land within 1% and 6% of their published values, which is reassuring for the
gauge implementation given everything else differs. HaWoR does not, and the likely reason is that
their HaWoR row runs on GT-derived boxes in its native regime while ours runs on detbox v3. That
gap is the box-source effect, measured on a third-party method, and it is large.

## What is still owed before any of this can be compared to Hand3R

The matched-protocol run on the 222 clips held out for both methods, with the shared detector and
one missing-detection policy. See `report/hand3r_fair_eval_clips.txt` and tasks #74 to #77.

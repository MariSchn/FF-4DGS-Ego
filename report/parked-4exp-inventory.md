# What is parked behind `\iffalse` in 4exp.tex

Audited 2026-08-18. `Sections/4exp.tex` has `\iffalse` at line 222 and `\fi` at line 1572, the
last line of the file. **1350 of 1572 lines do not compile.** The live section is lines 1-221.

This is intentional: a comment block at 215-221 is headed `STRIP BEFORE ANY OF THIS RETURNS`.
This document says what is actually in there, so the decision to revive or discard is made once
rather than one citation at a time.

## What is live today

| Line | Content | State |
|---|---|---|
| 4 | `\subsection{Experimental Setup}` | rewritten |
| 63 | `\subsection{Comparison with State of the Art}` | rewritten |
| 91 | `tab:camera` via `\input{tab_camera}` | **all `\tbd` except HaWoR H2O 264.1 / 71.7** |
| 97 | `\subsection{4D Gaussian Scene Reconstruction}` | rewritten |
| 113 | `tab:gs` | AnySplat + MoVieS rows filled |
| 198 | `\subsection{Analysis and Ablation Studies}` | rewritten |

The compiled bibliography holds 42 of the 89 bib entries. The other 47 are cited only from the
parked block, or not at all.

## What is parked

Nine tables, 36 `\todo` blocks, and the whole previous experiments section.

### Tables

| Label | Line | What it holds |
|---|---|---|
| `tab:datasets` | 328 | per-dataset pool composition |
| `tab:loss` | 751 | loss recipe |
| `tab:worldh3r` | 872 | world comparison, two-frame gauge |
| `tab:world` | 907 | world comparison |
| `tab:longwindow` | 1002 | the six-row world table (ours 35.4 / 26.5 / 27.0 / 70.7) |
| `tab:scale` | 1165 | **five-row scene-scale ablation, referenced from the LIVE appendix** |
| `tab:window` | 1447 | window-length sweep |
| `tab:boxsweep` | 1478 | box geometry sweep |
| `tab:abssup` | 1540 | absolute-supervision ablation |

`tab:scale` is the one that breaks the build: `6appendix.tex:157` cites it and the label does not
exist, so page 14 prints `??`.

### Positioning against the literature, all of it parked

| Line | Reference | What the paragraph does |
|---|---|---|
| 377 | ReViV | Tier 1, hands and scene in a world frame |
| 384 | HaWoR, Dyn-HaMR, EgoAllo | Tier 2, world hands without a scene |
| 393 | EgoForce | closest published absolute camera-frame placement |
| 415 | Human3R | the only general human-and-scene reconstructor |
| 428 | HGGT | defends the delta against a paper sharing our VGGT backbone |
| 437 | HaMeR, WiLoR, HaPTIC | Tier 3, camera-frame hands |
| 442 | several | what we do not run and why |
| 449 | - | the two comparability traps |

Losing this block loses the answer to "what is your delta against X", for every X.

### The 36 `\todo` blocks

They are not reminders, they are an audit trail. Four kinds:

1. **Corrections already applied**, recorded so they are not re-litigated: the camera-convention
   inversion (1011), the box-source audit (1080), row provenance for task #69 (1003).
2. **Contradictions between text and tables** that were never resolved: the protocol note at 276
   says the stated protocol is not the one behind the numbers, 607 says the same for the clip
   protocol, 1117 says the table predates the 30+100 protocol.
3. **Owed work**: a 21-joint both-hands C-MPJPE variant (547), the oracle-ceiling row (1073),
   the degenerate depth control on the full 157 (1499), leave-one-store-out (1340).
4. **Withdrawn claims** with the reason kept: 1289 withdraws a long-window claim, 270 records
   Re:InterHand's removal, 403 records cutting EgoForce.

## The decision this document exists to support

The live section is short, honest and mostly empty. The parked section is long, heavily audited
and full of numbers whose provenance was traced. Neither is submittable alone.

Reviving it wholesale re-imports every contradiction in category 2. Discarding it throws away
categories 1 and 4, which is exactly the material that stops a reviewer's objection from landing
twice. The order that follows from the audit trail is: settle category 2 first, since a table
that contradicts its own protocol paragraph is the one defect a reviewer cannot be talked out of.

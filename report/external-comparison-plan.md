# External SOTA comparison plan (the main-track make-or-break)

**Why this is now the gate:** with scene-metric falsified, the contribution is metric hand
placement. Main-track needs us to beat/match published SOTA (HaWoR, Hand3R) on a shared
benchmark. We currently only beat our teammate's checkpoint.

---

## The hardware blocker (and why it's not fatal)

This cluster is **Blackwell-only**: gb10 = GB10 (Grace-Blackwell), x86 jobs = RTX 5060 Ti (sm_120).
The SOTA stacks are older and **don't run on Blackwell**:
- **UniDepth** (E1 metric-depth FM): cu121 torch → "no kernel image"; cu128 fixes the arch but breaks
  UniDepth's API (needs torch ≤2.6 vs Blackwell's ≥2.7). **Conflict — blocked here.**
- **HaWoR**: torch 1.13 + cu117 (≤ sm_86 Ampere) + DROID-SLAM CUDA ext. Won't run on Blackwell without
  a torch→2.7/cu128 upgrade + a DROID-SLAM port — multi-day, uncertain.

**Not fatal:** these run fine on **Ampere (A100/A6000) or Hopper (H100)**. The fix is GPU *access*, not
a research blocker. One A100 (cloud or collaborator) unblocks the whole comparison.

---

## Concrete plan (priority order)

### Tier 1 — HaWoR on HOT3D (most tractable; do first)
HaWoR's repo **already ships `hot3d/scripts_eval/eval_hawor_hot3d.py`** and lists HOT3D sequences that
**overlap ours**. So the comparison is "run their eval," not "build one."
1. Get one Ampere/Hopper GPU (cloud A100 ~\$1–2/h, or a lab machine).
2. Install HaWoR there (conda py3.10 + torch1.13cu117 + DROID-SLAM build + HF weights — all in the
   README; weights already partially downloaded under `$HOME/HaWoR/weights`).
3. Run `eval_hawor_hot3d.py` on the shared HOT3D sequences → HaWoR's hand metrics + protocol.
4. Run **our** method on the **same** HOT3D sequences with HaWoR's metric (adapt `eval_hand_head` to
   match their MPJPE/world-frame protocol). Same clips, same metric.
5. Report ours vs HaWoR. **This is the single most important number for main-track.**

### Tier 2 — Hand3R on HOI4D (bigger; the strongest comparison)
Hand3R is the closest neighbor and reports on **HOI4D** (C-/W-MPJPE). This needs the **HOI4D port**:
1. HOI4D dataset access + preprocessing to our pinhole format.
2. Verify our backbone runs on HOI4D (the risk — different camera/scene).
3. Run our method + Hand3R's released eval on HOI4D. ~3–4 weeks.
This is the multi-week lift; it's what makes the comparison airtight for main-track.

### Tier 3 — Published-numbers comparison (fallback, weak)
If neither re-run is feasible in time, compare against HaWoR/Hand3R **reported** numbers on any
benchmark we can also evaluate on. Reviewers discount this (different splits/protocols), but it's
better than nothing for a rebuttal.

---

## What we need from you / decisions
- **GPU access**: a single A100/H100 (cloud or lab) for ≥ a few days unblocks Tier 1 immediately. This
  is the highest-leverage unblock. Can you get one?
- **HOI4D access**: needed for Tier 2 (the strongest comparison vs Hand3R). Apply now (access takes time).
- **Scope call**: Tier 1 (HaWoR/HOT3D) alone may suffice for a borderline main-track / strong workshop;
  Tier 1 + Tier 2 is a solid main-track. Tier 2 is the multi-week investment.

## Parallel: the unfreeze experiment (running, job 99405)
Independently, the unfreeze-encoder run tests whether a *true* metric scene is achievable (resurrecting
the headline). If it works (B2 object error drops below baseline 62cm), main-track gets much stronger and
the external comparison becomes supporting evidence rather than the sole pillar.

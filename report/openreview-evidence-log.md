# OpenReview Evidence Log

**Purpose.** Raw material for the reviewer-panel agent (task #55), and the standing reference for
every OpenReview-derived decision. Every attack the panel simulates must trace to a verbatim quote in
this file. Nothing here is paraphrased reviewer opinion presented as fact: quoted text is verbatim,
everything else is marked as our reading.

**Fidelity levels.** `FULL` = complete thread read, verbatim quotes recorded. `SUMMARY` = thread was
read in an earlier session and the conclusions survive, but verbatim text was lost to context
compaction and must be re-collected before the panel is built.

---

## 1. Calibration facts

These override intuition and must be encoded into the panel's AC agent.

| Fact | Evidence |
|---|---|
| **A unanimous 4/4/4 is a REJECT** | G-CUT3R. AC predicted all three would revise up (4→5, 4→5, 3→3-4) and still rejected. |
| The target is a genuine 6, not the absence of a hostile reviewer | 3 of 3 in-field samples rejected; the only 6 we saw (PhysHandi asRB) was outvoted |
| In-field base rate is harsh | 3 rejects from 3 topic-proximate ICLR 2026 samples, chosen by topic not outcome |
| Rebuttals routinely move scores | AC meta-reviews explicitly forecast score changes in both G-CUT3R and PhysHandi |
| **Not filing a rebuttal is fatal** | SIGHT: AC wrote "The authors did not submit a rebuttal" into the meta-review; two reviewers sat at 4 |
| Answering an attack is not the same as clearing it | G-CUT3R answered the efficiency attack with params + FPS and the AC still called it unresolved |
| Good writing does not save a weak contribution | SIGHT: presentation 2/3/3/2, "well-organized", "well written and easy to understand", contribution 1/1/2/1 |
| Losing on some metrics survives honest framing | SIGHT lost IV and CR in Tab. 1, wrote "four of six metrics ... favorably in the remaining two", no reviewer attacked it |
| Late-introduced favourable metrics are penalised | PhysHandi AC: "improvements rely heavily on newly introduced or task-specific evaluation choices" |
| A factually wrong review can still sink a paper | G-CUT3R iLL2 reviewed a hallucinated paper, AC acknowledged it, reject stood |

The single-agent `cvpr-reviewer` scores *published* papers at 10-12%, below our draft, so its
probabilities are defect rankings not forecasts. But its pessimism about the **bar** is closer to
right than we credited: see the base rate above.

---

## 2. Papers

### 2.1 SIGHT `FULL` - ICLR 2026 Submission 12271 - **Reject**

ETH (Gavryushin, Delitzas, Van Gool, Pollefeys, Mo, Xi Wang). Hand-object trajectory generation,
benchmarked on **HOI4D and H2O**, our exact two evaluation datasets. PDF read.

| | HFMJ | N5zj | bFJx | akaj |
|---|---|---|---|---|
| Rating | 2 | 4 | 4 | 2 |
| Confidence | 4 | 2 | 5 | 4 |
| Soundness | 2 | 3 | 3 | 1 |
| Presentation | 2 | 3 | 3 | 2 |
| Contribution | 1 | 1 | 2 | 1 |

**Attacks:**
- *Baseline relevance* (HFMJ, N5zj, bFJx independently). bFJx: "Why didn't you compare with Text2HOI? I believe this method was trained on the H2O dataset, and its results look realistic." N5zj: "the proposed method is only evaluated against an existing human body motion generation model, whereas there are many hand-object interaction generation models ... that are much more relevant."
- *Novelty by composition* (akaj): "this could be easily achieved by combining pre-existing text-to-hand motion generation (such as Text2HOI and DiffH2O) and 3D object reconstruction pipelines. I think authors need to empirically compare their method to these simple baselines."
- *Controlled benchmarks* (akaj): "The current datasets (HOI4D, H2O) are controlled benchmarks with known household-like manipulation actions. I think authors may have to include more in-the-wild samples."
- *Qualitative realism, watched frame by frame* (bFJx, confidence 5): "the wrist and object rotations are not aligned ... the bottle and wrist appear to rotate independently"; "the object moves as if magnetic forces are applied"; "The initial hand and object poses are not aligned with the input image."
- *Reviewer names the missing metric* (bFJx): "a metric measuring the alignment between the initial generated poses and the input image is needed. Since ground-truth motion data is available, you could measure the Chamfer Distance for the object, MPJPE for the hand, and relative distance and orientation errors."
- *Metrics do not measure the claim* (akaj): "Future trajectories could not be properly measured by the reported measures such as FID, diversity, etc."
- *Setup leaks the answer* (HFMJ): "since the input image (the hand and object are almost in contact) inherently leaks contact and affordance cues, the task becomes substantially easier."

**PDF-paired finding (unique to having both).** The baseline complaint was triggered by a
*one-clause* justification placed directly above Table 1: *"As the proposed SIGHT task is novel, we
adapt existing baselines from the whole-body motion generation literature."* The real rationale (that
Text2HOI/DiffH2O/MACS/CAMS/BimArt all require 3D object meshes as input) exists but sits four pages
earlier in Related Work. Reviewers reading the table never saw it.
→ **Rule: the reason a method is absent from the comparison table belongs next to the table.**

**Second PDF-paired finding.** Their exclusion rationale was self-defeating: they excluded
mesh-requiring methods because meshes are unavailable, but SIGHT-Fusion *retrieves a mesh* as an
intermediate step. N5zj: "Since the proposed framework obtains explicit 3D object geometry as an
intermediate representation ... comparing against a variant of the Text-to-HOI baselines with the
given 3D object geometry seems straightforward."
→ **Rule: check the exclusion rationale is not contradicted by your own pipeline.**

**Also observed:** they *do* have a Limitations paragraph. No reviewer credited it. Table stakes.

---

### 2.2 G-CUT3R `FULL` - ICLR 2026 Submission 16576 - **Reject** (unanimous 4/4/4)

Khafizov, Komarichev, Rakhimov, Wonka, Burnaev. **Architecturally our paper**: frozen encoder of a
pretrained feed-forward 3D reconstruction backbone (CUT3R), lightweight modality-specific encoders,
zero-conv fusion into the decoder, one model over arbitrary input subsets. Their sentence: *"Our
design also allows fusing directly to the decoder, allowing us to keep the whole encoder frozen."*

| | J8qk | iLL2 | bCZM |
|---|---|---|---|
| Rating | 4 | 4 | 4 |
| Confidence | 4 | 4 | 4 |
| Soundness | 2 | 3 | 3 |
| Presentation | 3 | 3 | 3 |
| Contribution | 2 | 1 | 3 |

**Attacks:**
- *Inherited efficiency* (J8qk) - **the one aimed straight at us**: "The paper claims to be efficient and lightweight. However, no additional results are provided to support this, such as FLOPs or parameter counts compared with CUT3R and Pow3R. Since the method introduces extra encoders and layers for additional modalities, it may incur substantial parameter and computation overhead relative to CUT3R, potentially compromising efficiency. **It is unclear whether the reported efficiency stems primarily from inheriting CUT3R's efficiency rather than being more efficient than CUT3R itself.**"
- *And answering it was not enough.* Rebuttal gave 463,438,546 → 532,819,599 params (+15%) and FPS 20 → 14-18. AC WfgJ still wrote: "concerns remain about ... whether the added modules materially increase compute and memory costs in common settings **beyond the reported FPS**, especially across different modality combinations."
- *Input provenance, with consequences attached* (bCZM): "It is a bit unclear to me what depth is used for the datasets. Did the authors always use sensor depth? **Using anything else would render the results incorrect.** I put this in the weaknesses given that this is very important." And: "Do the authors use SLAM-estimated poses (without post-processing) as priors or the GT ones? **Using the GT ones again would make the experiments section a bit weaker.**" Authors answered with a 12-row dataset / scene type / dynamic / depth source / pose source table.
- *Noise degradation curves* (bCZM): "Limited robustness analysis: no analysis of noisy/misaligned priors (e.g., wrong intrinsics, biased depth, pose drift). Results might depend strongly on prior quality. **I would be very happy to see plots showing how accuracy degrades with noise in the priors.**" Answered with a 5-50%-of-GT Gaussian noise sweep per modality, in Appendix E.
- *Limited conceptual novelty* (iLL2, and the AC's first-named remaining concern).
- *Hallucinated review* (iLL2): wrote a complete review of "Cross-View Uncertainty Tuning (CUT)" and "Geometry-Guided Regularization (G-Reg)", components that do not exist, apparently from parsing G-CUT3R as G + CUT + 3R. Scored Contribution 1. Authors: "Our work does not introduce or build upon CUT or G-Reg at any point. Many of the reviewer's comments refer to methods and contributions that are not present in our work." AC acknowledged it. Reject stood.
- *Backbone generality* (iLL2): "have the authors tested whether these regularizations still help when applied to other backbones (e.g., Fast3R or VGGT)?" Authors could not run it in the rebuttal window.

---

### 2.3 PhysHandi `FULL` - ICLR 2026 Submission 3336 - **Reject**

Jihyun Lee, Changmin Lee, Donghwan Kim, Tae-Kyun Kim. Hand + deformable object reconstruction.

| | rNHX | asRB | S4t5 | rJqj |
|---|---|---|---|---|
| Rating | 4 | 6 | 4 | 2 |
| Confidence | 3 | 4 | 4 | 5 |
| Soundness | 2 | 2 | 2 | 2 |
| Presentation | 2 | 3 | 3 | 2 |
| Contribution | 2 | 2 | 2 | 2 |

**Attacks:**
- *Title promises what the tables do not deliver* (rNHX) - **directly ours**: "This paper lacks experiments directly evaluating the accuracy of hand reconstructions. **The title of the paper gives hand and object the equal position, but only evaluate objects.** The claim in the Abstract ... is not fully supported."
- *Incremental over one prior work* (rNHX, asRB, S4t5, and the AC's headline): "multiple reviewers still view the method as an incremental extension of PhysTwin rather than a fundamentally new modeling or learning paradigm."
- *Only one baseline* (S4t5): "The paper only compares one method. Please provide comparisons with more methods on more datasets. For the qualitative comparison, please provide results from multiple methods on the same test samples."
- *Upstream input robustness* (asRB): "Can the authors include ablations on input quality (e.g., depth noise, missing data) and on dependencies (e.g., CoTracker accuracy, initialization errors) to quantify robustness?" (S4t5 asked the same for degraded hand estimates.) Answered with ~1mm depth noise, ~1px track noise, ~10mm MPJPE hand noise.
- *Controlled lab conditions* (rJqj, confidence 5, rating 2): "All experiments are conducted in highly controlled RGB-D capture setups ... There is no evidence that the method generalizes to in-the-wild scenarios."
- *Missing videos* (rJqj) - **the appendix driver**: "For a reconstruction method, more extensive qualitative evidence (especially videos in supplementary material) is crucial for evaluating realism and stability. **Without this, it is hard to be fully convinced of the claimed improvements.**"
- *Late favourable metric penalised* (AC zV4T): "Quantitative gains are modest on standard metrics, and improvements rely heavily on newly introduced or task-specific evaluation choices." (They introduced CD_dyn in rebuttal because CD_org showed a thin gap.)
- *Reframing attack* (rNHX): "this manuscript is more like a dynamic object simulation or cloth simulation paper with a hand-refinement module." A reviewer re-describing your paper as something lesser is a distinct failure mode from attacking a claim.

---

### 2.4 Accepts `SUMMARY` - verbatim lost to compaction, re-collect before building the panel

Human3R (ICLR 2026), JOSH, STream3R, Fin3R (NeurIPS 2025), Fuse-and-Refine, Trust3R, EgoHandICL,
OphNet-3D, MEgoHand. Surviving conclusions:

- Efficiency is the top attack. Human3R reviewer N5HW **ran the released code** and reported the
  inference speed "falls far short of real-time, even significantly below the 5 FPS reported in
  Table 4". Cost roughly four points and dominated the meta-review.
- A runtime breakdown was demanded in 5 of 6 threads examined; the repair that worked every time was
  a **per-component table** (Human3R Tab. 5-7, GVHMR/TRAM stage tables, Fuse-and-Refine's
  static/streaming split).
- Every accept carried at least one *survivable* defect: limited novelty, a missing ablation, losing
  to a baseline, or a 3 or 4 in the spread. None was clean.
- What separated the ~37% anchor from the ~10% ones was three cheap structural things: ablations
  mapping one-to-one onto stated contributions, an explicit limitations section with a failure
  figure, and one upstream input varied with the degradation on record.
- Fin3R: full fine-tuning is WORSE than frozen (26.35 vs 28.40 AUC@5). Load-bearing for our frozen
  backbone defence.
- JOSH reviewers asked twice for upstream-input degradation.

### 2.5 3D Affordance Reconstruction from Egocentric Demonstration Video `SUMMARY` - ICLR 2025 Withdrawn

Unanimous 3/3/3/3, soundness 1/2/1/2, presentation 1/1/1/2. The **catastrophic** cluster, distinct
from the borderline rejects above: no baselines at all + unverifiable evaluation + undefined metrics
+ technical-report prose. Quoted independently by all four: "No baseline comparisons are provided at
all" / "there is no comparison with other methods" / "The metrics used in experiments are not
explained properly" / "resembles a technical feasibility report more than a polished academic paper".
None of the nine accepts had this cluster. **We do not resemble this paper**; it defines the floor,
not the boundary.

---

## 3. Attack taxonomy, frequency-ranked

Counts are over the 3 FULL reject threads plus surviving conclusions from the accepts. This is the
list the panel agent's reviewer lenses are built from.

| # | Attack | Count | Exemplar | Our exposure | Task |
|---|---|---|---|---|---|
| 1 | **Efficiency claim unaudited / inherited** | 6+ | G-CUT3R J8qk; Human3R N5HW ran the code | HIGH. One FPS number, GS branch off, superseded T=16, detection excluded | #50 |
| 2 | **Nearest-neighbour baseline missing** | 5 | SIGHT x3, PhysHandi S4t5 | HIGH. No EgoGrasp row | #43/#44 |
| 3 | **Upstream-input degradation not swept** | 5 | G-CUT3R bCZM, PhysHandi asRB+S4t5, JOSH x2 | MED. Box jitter exists, not presented as a curve | #52 |
| 4 | **Controlled benchmarks, no in-the-wild** | 3 | SIGHT akaj names HOI4D+H2O by name | HIGH. Those are our two eval sets | mixing result |
| 5 | **Novelty by composition ("just combine X+Y")** | 3 | SIGHT akaj, PhysHandi AC | MED. We already beat HaMeR+SLAM / WiLoR+SLAM ~2x; foreground it | reframe |
| 6 | **Input provenance: GT vs estimated** | 2 | G-CUT3R bCZM, with consequences attached | HIGH. Hand-derived scale solve was previously *denied* in the text | #51 |
| 7 | **Qualitative evidence thin / no video** | 3 | PhysHandi rJqj (conf 5), SIGHT bFJx | HIGH. No supplementary at all | #53 |
| 8 | **Title/claims exceed what is evaluated** | 1 | PhysHandi rNHX | HIGH. WorldHand4DGS, GS never evaluated | #49 |
| 9 | **Metrics do not measure the claim** | 2 | SIGHT akaj; PhysHandi AC on late metrics | MED | #54 |
| 10 | **Incremental over one named prior work** | 4 | PhysHandi (PhysTwin) x4 | MED. Ours would be "NeoVerse/VGGT + a head" | reframe |
| 11 | Reviewer reviews the wrong paper | 1 | G-CUT3R iLL2 | LOW but unmitigable except by naming | #49 |

---

## 4. Standing rules derived

1. The reason a method is **absent** from a comparison table goes next to the table, not in Related Work.
2. An exclusion rationale must not be contradicted by our own pipeline.
3. Report FLOPs + params + peak memory across configurations, with inherited cost separated from added cost. FPS alone is proven insufficient.
4. Disclose GT-vs-estimated inputs per dataset in a table, including the hand-derived scale solve.
5. Present robustness as a degradation **curve** against a baseline, not a single row.
6. Lock the metric set before submission. A metric introduced late reads as engineered.
7. Losing on a metric is survivable if stated plainly in the reviewer's own arithmetic ("four of six").
8. Ship a supplementary video with the same clips across methods, plus a failure case.
9. Always file a rebuttal. Always.
10. A limitations section is table stakes, not a differentiator.

---

*Last updated 2026-08-05. Threads 2.1-2.3 read in full from OpenReview; 2.4-2.5 are surviving
conclusions pending verbatim re-collection.*

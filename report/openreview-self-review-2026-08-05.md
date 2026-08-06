# Simulated OpenReview: WorldHand4DGS, as of 2026-08-05

Four reviewers with distinct lenses drawn from measured archetypes in
`openreview-evidence-log.md`, plus an AC. Every attack below is one a real reviewer made of a
comparable submission, re-aimed at our draft. Scored on the paper as it stands, **assuming the
abstract, introduction and conclusion get written competently**, since all three are currently empty
and an actual submission in this state would be desk-rejected without review.

**Predicted scores: 3 / 3 / 2 / 4. Decision: Reject.**

---

## Official Review by Reviewer aQ7x (efficiency and cost lens)

**Summary.** The paper adds a hand branch to a frozen feed-forward reconstruction backbone
(NeoVerse/VGGT), claiming absolute metric hand pose plus 4D Gaussian scene reconstruction in one
pass, and positions itself against SLAM-composed per-frame estimators on efficiency grounds.

**Soundness 2 · Presentation 3 · Contribution 2 · Rating 3 · Confidence 4**

**Strengths.** Parameter efficiency is a legitimate axis on which to compete, and the paper does
report both a parameter count and a throughput figure rather than hand-waving.

**Weaknesses.**

1. **The parameter claim and the speed claim come from two different networks in the same table.**
   The paper states 46.3M trainable of 1178.7M (3.9%), but by its own admission the probe producing
   the 6.59 FPS cell builds the model with the Gaussian head present and reports 262.4M trainable
   (22.3%). One of these two numbers describes the system the paper claims to contribute; they
   cannot both.
2. **The efficiency headline is purchased by switching off the second claimed contribution.** 3.9%
   is 3.9% only because the Gaussian branch is disabled. The paper claims hand-to-scene feature
   injection as a contribution, then reports its efficiency with that pathway off.
3. **It is unclear whether the reported efficiency is inherited or earned.** The backbone is frozen
   and pretrained. No FLOPs, no peak memory, no per-component attribution separating what the
   backbone costs from what the hand branch adds. Reporting a single FPS figure cannot distinguish
   an efficient contribution from an efficient inheritance.
4. **The 6.59 FPS excludes detection** while the 2.47 FPS comparison figure includes it, and it is
   measured at a clip length the paper elsewhere says is superseded. The caption discloses this,
   which is to the authors' credit, but disclosure of a non-comparable ratio does not make it
   comparable.

**Questions.** What are the FLOPs and peak memory for backbone-only, backbone plus hand branch, and
the full system? What is the trainable parameter count of the configuration you are actually
proposing? Which clip length backs the throughput number?

---

## Official Review by Reviewer mK4d (protocol and provenance lens)

**Summary.** As above. My concerns are with whether the reported comparisons measure what they claim.

**Soundness 2 · Presentation 2 · Contribution 2 · Rating 3 · Confidence 4**

**Strengths.** The input-matching discussion is more careful than is typical, and the authors
volunteer a detector-box protocol rather than quietly using ground-truth boxes.

**Weaknesses.**

1. **The main long-window table is not input-consistent, and the paper says so.** Three rows resolve
   to the authors' detector boxes and three to each method's own detector. The Input Matching
   paragraph claims to have removed exactly this confound. A table that violates its own stated
   protocol cannot support a ranking.
2. **The authors' own rows cover 60 of 157 sequences** (180 segments) while the baselines cover 468.
   These are not the same evaluation.
3. **The model is trained on the target dataset and the baselines are not.** The reported HOI4D
   numbers come from a checkpoint trained on HOI4D-367 and evaluated on HOI4D-157. Every runnable
   baseline is cross-dataset. The paper's own Eq. 5 identifies train-test depth shift as the axis
   governing absolute error, so the comparison is confounded along precisely the axis the paper
   theorises about.
4. **A scale is solved from the prediction and applied to every world number.** The world lift
   multiplies camera translation by the median ratio of predicted hand depth to predicted scene
   depth, recorded at 0.707 against a true scale near 1.0. This must be disclosed per dataset with
   the depth and pose sources, and the paper must state which of the three scale variants produced
   each tabulated number. **Using a hand-derived scale where a metric scene is claimed materially
   weakens the metric claim.**
5. **The flagship analysis is confounded by that same scale.** The GT-camera oracle taking W-MPJPE
   from 200.9 to 61.5 substitutes rotation, translation and a roughly 29% scale error
   simultaneously. "Long-window error is camera-dominated" does not follow from a three-way
   substitution. A decomposition is required.
6. **The described protocol is not the protocol behind the numbers.** The paper describes T=32 with
   n~U{2..32}; the tables come from a T=16 fixed-length model. Likewise the dataset section
   describes holding out both H2O and HOI4D, which is not what produced any reported result.

**Questions.** Will the tables be rebuilt box-consistently at full sequence coverage? Which scale
variant backs each world number? Can you provide the GT-scale-only / GT-rotation-only / GT-full
decomposition?

---

## Official Review by Reviewer tR9v (contribution and framing lens)

**Summary.** Three contributions are claimed: scene features improve hand estimation, hand features
improve scene reconstruction, and both are placed at a common metric scale.

**Soundness 2 · Presentation 3 · Contribution 1 · Rating 2 · Confidence 4**

**Strengths.** The cross-dataset depth-bracketing result is genuinely interesting, the prediction was
registered before the run, and 184.8 to 66.2 zero-shot is a large effect. The absolute-supervision
2x2 is a clean causal design.

**Weaknesses.**

1. **The second contribution has never been run.** By the authors' own annotation, every
   configuration behind every reported number sets `enable_gs: false`. The injection convolutions
   receive no gradient. The described design and the code disagree about what is frozen. A
   contribution that has not been trained or evaluated is not a contribution.
2. **The first contribution is refuted by the authors' own ablation.** A frozen generic DINOv2
   backbone reaches 21.9 against the reconstruction backbone's 23.6. If reconstruction features are
   not what helps, "enhancing hand estimation with scene features" is not what the result
   demonstrates. The paper acknowledges this and does not resolve it.
3. **The third contribution reduces to a loss weight.** The 2x2 shows absolute supervision as a class
   is necessary but that its particular form is interchangeable (27.1 or 23.0 against 22.8). "Apply
   an absolute keypoint loss" is a training recipe, not a method.
4. **The method loses to its own control on the headline metric.** A fine-tuned HaMeR on identical
   boxes with identical augmentation reaches 23.4 against 35.6, and degrades less under detector
   boxes. The paper reframes this as a parameter and latency trade, but a reviewer must ask what the
   paper contributes if a simpler, older architecture with the authors' own recipe is 34% more
   accurate.
5. **The title claims what the tables do not deliver.** The paper is titled around joint 4D Gaussian
   reconstruction and hand tracking. No Gaussian reconstruction quality is evaluated anywhere. Given
   weakness 1, the scene half of the title corresponds to no experiment at all.

**Questions.** With the DINOv2 result in hand, what is the claim about scene features? What would the
paper claim if the fine-tuned HaMeR row is accepted as the correct comparison?

---

## Official Review by Reviewer zB2p (hand pose domain expert)

**Summary.** As above. I am familiar with HaWoR, HaMeR, WiLoR, Hand3R and the HOI4D/H2O benchmarks.

**Soundness 2 · Presentation 3 · Contribution 2 · Rating 4 · Confidence 5**

**Strengths.** The baseline suite is unusually thorough for this area and several baselines were run
rather than reprinted. The honesty is real: the authors report a reversal that cost them their
headline, a null result that refutes their own hypothesis, and a control they lose to. I would
rather review this than most papers in this area. The depth-bracketing analysis is the most novel
thing here and is under-sold relative to the architectural claims.

**Weaknesses.**

1. **The headline metric uses a systematically favourable joint set.** Camera-frame numbers are
   right-hand-only over 16 joints excluding fingertips. HaMeR, WiLoR, HaWoR and Hand3R report 21
   joints including fingertips, which are the highest-error joints on a hand. The paper states this
   and does not correct it. Every cross-paper comparison in the tables is therefore biased in the
   authors' favour by an unquantified amount. A 21-joint variant is required before I can read the
   comparison at all.
2. **A segment-rigid error is bolded as a global metric.** The 37.3 W-MPJPE is computed with a
   30-frame alignment window on 30-frame segments, so no drift is accumulated by construction. The
   authors identify this in their own text. At 128 frames the same quantity is 200.9. Presenting
   37.3 as a world-space win is not defensible.
3. **Losing 166mm between W and WA at 128 frames** indicates the world-frame trajectory is not usable
   at length. The paper is honest about this and calls it a characterised limitation, but the title
   and framing promise world-space reconstruction.
4. **No supplementary video.** For a method whose central claim is temporal placement in a world
   frame, still figures cannot show drift or jitter. Without video I cannot verify the qualitative
   claims. The existing qualitative figure is drawn from checkpoints predating the current recipe and
   from three unrelated sequences, per the authors' own note.
5. **No failure case analysis** and no comparison figure against HaWoR or WiLoR in the same scene at
   metric depth, which is the picture the central claim invites.

**Questions.** Can you provide 21-joint both-hand numbers? Can you provide video? What does the model
do when hands leave the frame or when the scene is texture-poor?

---

## Meta Review by Area Chair

**Summary.** The paper adds a trained hand branch to a frozen feed-forward reconstruction backbone
and targets absolute metric hand pose in a world frame from egocentric video. Reviewers agree the
problem is well-motivated and that the empirical work is unusually candid.

**Reviewer concerns.**

Addressable with revision: the joint-set mismatch, the missing FLOPs and memory accounting, the
absent video and failure analysis, and the input provenance table.

Not addressable within a revision cycle: three reviewers independently observe that one of the three
claimed contributions has never been trained or evaluated, that a second is contradicted by the
authors' own backbone ablation, and that the method is outperformed on its headline metric by a
fine-tuned single-image baseline under matched inputs. tR9v further notes the title corresponds to an
experiment that does not exist. mK4d establishes that the primary long-window table is not
input-consistent and that the authors' own rows cover roughly a third of the evaluation set, so the
central ranking is not currently supported.

The candour is noted and appreciated by all four reviewers. It does not substitute for the missing
experiments.

**Reviewer scores.** 3, 3, 2, 4. No reviewer recommends acceptance.

**Decision: Reject.**

---

## Calibration and probability

Anchored against measured outcomes rather than intuition:

| Paper | Scores | Mean | Outcome |
|---|---|---|---|
| G-CUT3R | 4/4/4 | 4.0 | **Reject** (AC expected upward revision) |
| PhysHandi | 4/6/4/2 | 4.0 | **Reject** |
| SIGHT | 2/4/4/2 | 3.0 | **Reject** |
| **ours, today** | **3/3/2/4** | **3.0** | — |

We score at the SIGHT line, and a full point below two papers that were rejected. Since a unanimous
4/4/4 did not clear the bar, **acceptance probability in the current state is under 5%.** That figure
is a ranking against these four anchors, not a forecast, and it should be read as "materially worse
than three papers that were all rejected."

The reachable ceiling is different. Most of what sank this review is fixable in six weeks, and two of
the four reviewers would move substantially on evidence that already exists or is cheap to produce.

## Flip list, ranked by score moved per unit cost

1. **Run the GS branch once, or cut the contribution and retitle.** Removes tR9v's weakness 1 and
   weakness 5 together, and fixes the parameter/FPS inconsistency that drives aQ7x. Single highest
   leverage item in the paper. If it will not run, cutting it honestly is worth more than claiming it.
2. **Rebuild the long-window table box-consistently at full 157-sequence coverage.** mK4d cannot
   recommend anything until the central table supports its own ranking.
3. **21-joint both-hand variant for every cross-paper row.** zB2p, the highest-confidence reviewer,
   blocks on this. It is a scorer change, not a retrain.
4. **Supplementary video plus a failure figure.** Cheap, and answers zB2p's weaknesses 4 and 5.
5. **FLOPs, memory, per-component attribution, on one consistent configuration.** Answers aQ7x
   entirely.
6. **Scale decomposition (GT-scale / GT-rotation / GT-full) and the provenance table.** Converts
   mK4d's weaknesses 4 and 5 from objections into contributions.
7. **Reframe around depth-diverse mixing.** zB2p already flags it as the most novel result and says
   it is under-sold. It is the one finding here with a registered prediction that held, and it
   survives every attack above because it does not depend on the architecture.

Items 1, 3, 4 and 7 need no new GPU results. Item 2 is a re-run of existing evaluations.

## The uncomfortable read

The paper currently claims an architecture and demonstrates a training recipe. Its three strongest
verified results, the absolute-supervision 2x2, the depth-bracketing transfer law, and the box-cue
null, are all statements about supervision and data rather than about the scene-conditioning
architecture the title and Method sell. The reframe in item 7 is not damage control; it is aligning
the claims with what the evidence actually shows, which is the single thing that separates the
rejected papers we studied from the accepted ones.

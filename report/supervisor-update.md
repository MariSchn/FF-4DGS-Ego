Hey Chenyangguang, quick update on where I got since the report/poster.

I kept pushing on the metric side. Two things, and the first one I should explain since we didn't
really discuss it.

First, the hand placement. In the report we only looked at PA-MPJPE, which lines the hand up first, so
it tells you the shape is good but not whether the hand is actually in the right place in 3D. I wanted
to check the placement itself, so I added a loss on the absolute 3D joint positions (not pelvis aligned)
and measured the absolute MPJPE. That got it down from 81mm to 53mm on a robust 9 seq split (-35%), with
the hand depth around 4.5cm. So the hand sits in the right place metrically now, not just the right shape.

To check the hand is actually a good metric anchor, I compared it against UniDepth-V2, a current metric
depth foundation model: at the hand it lands about 15.7cm off, vs 4.5cm for our in-scene metric hand, so
the hand is roughly 3.5x better as a scale anchor than an off the shelf depth model. That's the justification
for anchoring the scene scale to the hand in the first place.

Second, the scene. I built the region masked object evaluation we said was the next step in the report.
I render the GT object depth from the HOT3D object meshes, mask out the hand, and compare the predicted
depth on the objects (sanity checked it: where the hand touches an object the GT render matches the metric
hand to about 2 to 4cm). The honest result is the scene depth does not become metric on the objects. When
I push the metric loss the predicted depth on objects is around 135cm off, vs about 62cm without it, so it
actually gets worse.

The earlier "the scale is getting more consistent" signal I had was misleading, because I was measuring it
at the hand, where the model is trained to match the metric hand, so of course it looks good there. On the
objects it isn't.

The reason is the frozen backbone. Even with no metric loss its scene depth is already around 62cm off on
the objects, so the metric hand can only fix the global scale, not the relative structure. Pushing the
metric loss on top of a frozen backbone just distorts it. The reconstruction quality stays fine btw (PSNR
within about 0.3dB of the baseline), so we don't lose the rendering gain, it's specifically the metric
depth that doesn't hold.

So I think the real next step is what you and Marian suggested, train the Gaussian head and the backbone
on the metric depth from HOT3D instead of keeping the backbone frozen. I started that, and the early signal
is encouraging: with the backbone unfrozen the validation C-MPJPE drops from 256mm to about 73mm in the first
150 steps. The problem is it doesn't survive, training the full backbone at 224x224 on the single consumer
card runs out of memory and the run dies after a couple of hours, before it converges or reaches the object
depth eval. So I can't actually finish this run on the current hardware. If the object depth drops below the
62cm baseline then we have the unified metric model (Gaussians, depth, camera, hand in one pass), which is the
thing that puts us ahead of the 2 stream setups. The scale head route you mentioned is also on the table as a
simpler, lighter version that should fit in memory.

The thing slowing this down a lot is the GPU. I'm on a single consumer card, so I'm stuck at 224x224, batch
around 2, and it's very slow (a training run is hours, and training the backbone makes it worse, that's the
run that keeps OOMing). The UniDepth baseline I mentioned also won't run on the Blackwell cards I mostly have
(too new for its cuda), I had to find an older Turing card just to get that one number. To iterate properly on
the metric depth training I'd really need a better GPU. Any chance I could get access to an H100 or A100 for a
bit? Honestly that's the main bottleneck right now between the current results and pushing this towards a
publication.

I'm attaching a comparison table that pulls all of this together: the hand-anchor vs depth-FM result, the
absolute placement gain (81 to 53mm), the HOI4D dense-depth number (scene depth residual 20.5 to 13.4cm with
no masking), and where the reported world-space methods (HaWoR, Hand3R) sit. The two scale experiments we
discussed are mid-flight right now: I'm training both the partial-unfreeze with direct GT object-depth
supervision and the lighter scale head you suggested. The world-space rows fill in as those converge; the one
thing slowing them is GPU/disk (training a backbone on the single consumer card, plus the scratch quota), so
converged numbers come a lot faster on a bigger card.

Happy to go through the numbers whenever. I can have the clean ablation (how much the absolute 3D loss adds
over plain finetuning) shortly; the full backbone training result really depends on getting a card that can
hold it, otherwise I only get the first few hundred steps before it dies.

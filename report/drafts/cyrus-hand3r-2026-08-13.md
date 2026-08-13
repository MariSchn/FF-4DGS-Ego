# Message for Cyrus, 2026-08-13

He has already read their email, so this says only what he cannot know from it: what I measured
after receiving the split, and the three decisions.

---

Hi Cyrus,

I checked our split against theirs. 67.5% of our 157 test sequences are inside Hand3R's training
set, and a third of their 300 evaluation clips are inside ours. Neither split works. The
intersection held out for both is 222 clips. I have the ground truth for all 222 and am building
the store now.

I also re-scored every row under their W gauge, which fits the transform on the first two frames
where ours fits thirty. It roughly doubles every global number we have ever reported, ours
included, but it costs us less than any baseline (+41mm against +56 to +86), so our lead widens
rather than shrinks. Our short-video margin over the best baseline goes from 8.3mm to 23.8mm.

Where we stand on our own split under their gauge: 35.4 C-MPJPE, 28.1 / 73.0 short, 43.3 / 121.2
long. We beat every offline baseline including HaWoR. Their paper attributes its own loss to HaWoR
to the online-versus-offline distinction and calls drift inevitable for causal methods, so if this
holds on the matched set it contradicts that, which is worth more than the margin.

Three decisions:

1. Accept their offer to run Hand3R? It means sending them our boxes, format and scorer.
2. HOI4D evaluation on the 222 clips, or report both sets?
3. Adopt their W gauge as primary?

Dario

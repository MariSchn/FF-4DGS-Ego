# Message for Cyrus, 2026-08-13

Hi Cyrus,

The Hand3R authors replied. No code or checkpoints, but they sent their full evaluation protocol,
a reference scorer, and their exact HOI4D split, and they offered to run Hand3R themselves under
our protocol and send back predictions.

Two things changed because of the split. Hand3R fine-tunes on 75% of HOI4D, and 67.5% of our
157-sequence test set sits inside their training data. Going the other way, a third of their
evaluation clips are in our training set. So neither split can carry a comparison. The intersection
that is held out for both is 222 clips, and that is what I propose we evaluate on.

Their W-MPJPE also aligns on the first two frames of each chunk where ours aligns on the first
thirty. I re-scored every row under both. Their gauge roughly doubles all global numbers, ours
included, but it costs us less than any baseline, so our lead actually widens. I think we should
adopt it and report both.

Current numbers on our own split, using their gauge, short and long video:
we get 35.4 C-MPJPE, 28.1 / 73.0 short, 43.3 / 121.2 long, and we beat every offline baseline
including HaWoR. Their paper says the online-vs-offline gap is structural, so if that holds under
the matched protocol it is a real result.

One blocker: HOI4D hand-pose ground truth is not in the public mirror, only on OneDrive, and I
cannot download it programmatically. We have it for our 524 sequences and none of the 222.

Decisions I need from you:

1. Should we accept their offer to run Hand3R? It means sending them our boxes, format and scorer.
2. Do we move the HOI4D evaluation to the 222 clips, or report both sets?
3. Do we adopt their W gauge as the primary one?

Dario

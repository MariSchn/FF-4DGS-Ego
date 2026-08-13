# Reply to Wendi (Hand3R), draft 2026-08-13

Keep it short. They wrote a long, generous email; the useful reply is the one that shows we
actually used what they sent and gives them something concrete to say yes to.

---

Hi Wendi,

Thank you, this was more than we hoped for, and the split changed our plan.

We checked our HOI4D split against yours straight away. 106 of our 157 evaluation sequences,
so 67.5%, are in your training set, and going the other way, 103 of your 300 paper clips are in
ours. So neither split can carry the comparison, and we would rather not publish a number that
flatters either of us.

What does work is the intersection. Taking your held-out set, removing anything we trained on,
and then also removing anything sharing an (H, C, S) scene with our training data, leaves 222
clips over 26 scenes that are genuinely unseen for both models. We are downloading and converting
those now. Our current data does not cover any of them, which is the point.

We would very much like to take you up on running Hand3R on it. We will send the clip list, our
detector boxes, and predictions in exactly the NPZ layout your scorer already reads, so you can
score it with your own code. We will also send our scorer so you can check we run yours unmodified.

Two things we will label clearly, following your advice. The benchmark uses a shared detector
rather than GT-derived boxes, so it is a new input protocol and not a reproduction of Table II.
And we will fix a single missing-detection policy for every method before we run anything.

One correction on our side that your PROTOCOL.md caught: our W-MPJPE fits the rigid transform on
the first 30 frames, not the first two. We had assumed those matched. We have implemented your
version and will report both.

Happy to send everything for review before we submit.

Best,
Dario

---

## Checks before sending

- [ ] 222-clip list attached (`report/hand3r_fair_eval_clips.txt`)
- [ ] missing-detection policy decided, in one sentence
- [ ] Cyrus has read it

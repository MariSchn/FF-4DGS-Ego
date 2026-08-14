# For Cyrus: can we use the full five, plus two things from the citation sweep

---

**Can we use the full set of five?** Only OakInk2 is worth it. The other four are already at their
useful or possible maximum.

- **ARCTIC**: using all 339 means training on their test split. I would keep the 267 train split.
- **HOT3D**: 136 is everything with public ground truth. Quest 3 has no RGB, our pipeline cannot use it.
- **TACO**: the 6 missing have no ego video in the release.
- **DexYCB**: 8 cameras instead of 3 gives the same 1,000 captures from more angles of a fixed rig
  with no egocentric view, so no new motion and no new viewpoint. Costs ~500 GB of cache we do not have.
- **OakInk2**: this one is real. 109 of 627, one scene of four, because the download died. Worth
  redoing, and the converted store is only ~25 GB.

---

**Two corrections to our bibliography.** We cite NeoVerse as an arXiv preprint, but it is CVPR 2026
Highlight. Same for VGGT, which is CVPR 2025. We are citing our own backbone as unpublished.

---

**One paper you should see.** Hand-4DGS, arXiv 2606.19156, June 2026, from Bae, Kim, Pollefeys,
Rad, Uh and Kwon. Feed-forward Gaussians, egocentric, 4D hands, evaluated on H2O and ARCTIC. That
is our title, our modality and two of our datasets, and Kwon is an H2O author.

It is concurrent by ICLR timing and unpublished, so by the rule we just applied to Hand3R we do not
have to compare. But a reviewer will find it, so we should read it and decide what to say before
someone else decides for us.

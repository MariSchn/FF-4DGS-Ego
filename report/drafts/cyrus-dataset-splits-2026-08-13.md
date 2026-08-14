# For Cyrus, about what 4.1 can claim

---

I checked what we actually train on, because 4.1 says complete datasets. Only TACO is.

- **ARCTIC**: 267 of 339, which is exactly the official p2 train split
- **DexYCB**: all 1,000 captures, but only 3 of 8 cameras
- **HOT3D**: 136 of 198 Aria recordings, the rest have no public ground truth. Quest 3 not downloaded
- **OakInk2**: 109 of 627, one of four scenes. The download was killed partway
- **TACO**: 2,311 of 2,317, the six missing have no ego video

OakInk2 needs a decision. It is not a sample, the download stopped inside scene_01, and all 627 are
public, so we cannot call ours the released subset. Re-download or say plainly it is a fraction of
one scene.

How specific should 4.1 be? One line per dataset with the counts, or one sentence saying official
splits where they exist and a subset otherwise with the numbers in the appendix?

I would give the counts for all five. ARCTIC, HOT3D and TACO have respectable reasons, and hiding
DexYCB's three cameras or OakInk2's fraction in a total is the kind of thing a reviewer finds.

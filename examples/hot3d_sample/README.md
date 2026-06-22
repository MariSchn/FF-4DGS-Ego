# HOT3D sample (the egocentric data we use)

This project trains and evaluates the hand head on HOT3D (Meta Project Aria,
egocentric hand-object recordings). That is the sample data we used.

We do not commit HOT3D imagery to this public repository. HOT3D is released under a
license that requires accepting an agreement to download and restricts
redistribution, so raw frames, and renders that show them, are kept out of the
public repo.

How to get the data:

- Dataset: <https://www.projectaria.com/datasets/hot3d/>
- Tooling: <https://github.com/facebookresearch/hot3d>

We preprocess each sequence to undistorted pinhole frames (focal 609) with hand and
camera ground truth. The model consumes 16-frame clips at 224x224.

A short real preprocessed clip from one HOT3D sequence is included separately in the
course submission package, not here, for the license reason above.

"""Tests for the opt-in variable-length training mode (``data.random_frames``).

CPU-only, no cluster, no real video/backbone. The dataset is instantiated without running
__init__ (which walks sequence directories) and fed a synthetic clip whose every per-frame
tensor is FRAME-ID STAMPED: element [i, ...] == i. That makes the property under test
directly observable - after subsampling, the stamp recovered from EVERY per-frame tensor must
equal the same sampled index vector. A tensor sliced with a different subset, or not sliced at
all, shows up immediately instead of degrading a metric quietly.

Covers:
  (a) key absent  -> output identical to the fixed-length path, no frame_index emitted
  (b) key present -> every per-frame tensor sliced with the SAME indices, per-clip keys untouched
  (c) n stays in [min, max], is shared by every sample of a batch, and is seed-deterministic
  (d) build_views timestamps carry the TRUE window indices, not 0..n-1
plus the config validator, the MixedHandDataset delegation, the forgot-to-register guard, and
a collate round-trip proving the shared-n design is what makes batching possible at all.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

# Same as conftest.py: make the repo root importable so `scripts.train_hand_head` resolves
# whether this file is run via pytest from the root or directly as a script from tests/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ------------------------------------------------------------------
# Import scripts.train_hand_head in a bare CPU sandbox.
# Same philosophy as tests/conftest.py: try the REAL import first (on the cluster every
# dependency is present and this loop exits on the first iteration), and only stub the
# modules that are actually missing. Stubs are minimal - the code under test never calls
# into them, they only have to survive being imported.
#
# One wrinkle: conftest.py installs attribute-less ``diffsynth`` package shims whenever the
# heavy import fails, and train_hand_head's import chain runs
# ``from ..auxiliary_models import WorldMirror``, which needs the package __init__ to have
# actually executed. So the shims are lifted out for the duration of this import and put back
# byte-for-byte afterwards, leaving the rest of the suite exactly the environment it expects.
# ------------------------------------------------------------------

_SHIMMED_PKG = "diffsynth"


def _install_stub(name: str) -> None:
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__file__ = f"<stub:{name}>"
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)

    def _getattr(attr: str):
        # Dunders must still raise: transformers/torch introspect __file__, __spec__ etc.
        # and a truthy answer for those sends them down completely wrong code paths.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return type(attr, (), {})

    mod.__getattr__ = _getattr
    sys.modules[name] = mod


def _is_pkg(name: str) -> bool:
    return name == _SHIMMED_PKG or name.startswith(_SHIMMED_PKG + ".")


def _repair_stub_specs() -> None:
    """Give every attribute-only stub module in sys.modules a __spec__.

    Sibling test modules (test_hand_pointcloud_3d, test_quat_axis_angle_grad) register bare
    ModuleType stubs for decord/cv2. transformers probes those with
    importlib.util.find_spec, which raises ValueError - not ImportError - when __spec__ is
    None, so the stub-on-demand loop below would never get the chance to fix it. Only
    file-less, spec-less modules are touched, i.e. only stubs.
    """
    for name, mod in list(sys.modules.items()):
        if mod is None or name == "__main__":
            continue
        if getattr(mod, "__spec__", None) is None and not hasattr(mod, "__file__"):
            mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)


def _import_train_module():
    _repair_stub_specs()
    saved = {k: v for k, v in sys.modules.items() if _is_pkg(k)}
    for k in saved:
        del sys.modules[k]
    stubbed: list[str] = []
    try:
        for _ in range(40):
            try:
                return importlib.import_module("scripts.train_hand_head")
            except ModuleNotFoundError as exc:
                if not exc.name or exc.name in stubbed:
                    raise
                # Never stub the code under test. Without this the loop would happily hand
                # back a stub `scripts.train_hand_head` whose __getattr__ invents a
                # HOT3DHandDataset, and every assertion below would be testing nothing.
                if exc.name == "scripts" or exc.name.startswith("scripts."):
                    raise
                _install_stub(exc.name)
                stubbed.append(exc.name)
        raise RuntimeError(f"could not import scripts.train_hand_head (stubbed: {stubbed})")
    finally:
        # The imported module object keeps its own references to everything it needs, so
        # restoring sys.modules here costs nothing and keeps this file side-effect free.
        for k in [k for k in sys.modules if _is_pkg(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


T = _import_train_module()

NUM_FRAMES = 16          # the cached window length the fake clip was built at
TOKEN_P, TOKEN_C = 3, 5  # tiny stand-ins for the real [S, P, C] token cache
OBJ_RES = 4


# ------------------------------------------------------------------
# Synthetic clip helpers
# ------------------------------------------------------------------

def _stamped(n: int, *tail: int) -> torch.Tensor:
    """[n, *tail] float tensor where every element of frame i equals i."""
    v = torch.arange(n, dtype=torch.float32).reshape(n, *([1] * len(tail)))
    return v.expand(n, *tail).clone()


def _parity(n: int, *tail: int) -> torch.Tensor:
    """[n, *tail] bool tensor that is True on even frames. Bools cannot carry the frame id,
    so the even/odd pattern is what proves they were sliced with the same index vector."""
    v = (torch.arange(n) % 2 == 0).reshape(n, *([1] * len(tail)))
    return v.expand(n, *tail).clone()


def _recover(tensor: torch.Tensor) -> list[int]:
    """Frame ids a stamped tensor is carrying, one per leading-axis entry."""
    flat = tensor.reshape(tensor.shape[0], -1).float()
    assert (flat == flat[:, :1]).all(), "stamped tensor is not constant within a frame"
    return [int(round(x)) for x in flat[:, 0].tolist()]


def _make_dataset(tmpdir: str):
    """A HOT3DHandDataset carrying exactly one fully-populated synthetic clip.

    __init__ is bypassed on purpose: it scans sequence dirs, decodes video and runs MANO.
    Only the attributes __getitem__ reads are set, so this exercises the real __getitem__
    (including the real feature-cache torch.load) and nothing else.
    """
    ds = T.HOT3DHandDataset.__new__(T.HOT3DHandDataset)
    ds.num_frames = NUM_FRAMES
    ds.res = (4, 4)
    ds.use_hand_crop = True
    ds.bbox_perturb = None
    ds.emit_cache_key = True
    ds.feature_cache_dir = tmpdir
    ds.render_obj_depth = True
    ds.obj_render_res = OBJ_RES
    ds.mano_model = None
    ds.clips = [{
        "video_path":     "<fake>",
        "seq_path":       "/fake/seq_A",
        "frame_offset":   7,
        "n_video":        100,
        "has_mano":       True,
        "gt_frames":      list(_stamped(NUM_FRAMES, 64)),
        "gt_joints":      _stamped(NUM_FRAMES, 2, 16, 3),
        "hand_bboxes":    list(_stamped(NUM_FRAMES, 2, 4)),
        "hand_valid":     list(_parity(NUM_FRAMES, 2)),
        "gt_joints_2d":   _stamped(NUM_FRAMES, 2, 16, 3),
        "cam_extrinsics": _stamped(NUM_FRAMES, 4, 4),
        "cam_intrinsics": torch.tensor([215.5, 112.0, 112.0]),  # [3], per CLIP not per frame
        "contact":        _parity(NUM_FRAMES, 2),
        "da3_wrist":      _stamped(NUM_FRAMES, 2),
    }]
    # Object depth is rendered from raw HOT3D meshes in the real dataset; stub the renderer
    # so the two keys it emits still go through the slicing path under test.
    ds._render_clip_obj_depth = lambda clip: (_stamped(NUM_FRAMES, OBJ_RES, OBJ_RES),
                                              _parity(NUM_FRAMES, OBJ_RES, OBJ_RES))

    # The frozen-feature cache is read with a real torch.load, so write a real file.
    tokens = _stamped(NUM_FRAMES, TOKEN_P, TOKEN_C).to(torch.bfloat16)  # [S, P, C], frame axis FIRST
    torch.save(tokens, os.path.join(tmpdir, "seq_A_7.pt"))
    return ds


@pytest.fixture()
def dataset(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="random_frames_test_")
    # Video decode and the PIL->tensor conversion are irrelevant to frame subsetting; replace
    # them with a stamped stand-in so `img` carries a frame id like every other tensor.
    monkeypatch.setattr(T, "VideoReader", lambda path: None, raising=True)
    monkeypatch.setattr(
        T, "load_video",
        lambda reader, num_frames, resolution, sampling, frame_offset: list(
            _stamped(num_frames, 3, 4, 4)),
        raising=True)
    monkeypatch.setattr(T, "TVF", types.SimpleNamespace(to_tensor=lambda x: x), raising=True)
    return _make_dataset(tmpdir)


# Keys the synthetic clip makes the dataset emit, split the way the code under test splits them.
EXPECTED_FLOAT_FRAME_KEYS = ("img", "gt", "gt_joints", "hand_bboxes", "gt_joints_2d",
                             "cam_extrinsics", "da3_wrist", "cached_tokens", "gt_obj_depth")
EXPECTED_BOOL_FRAME_KEYS = ("hand_valid", "contact", "gt_obj_mask")
EXPECTED_CLIP_KEYS = ("cam_intrinsics", "has_mano", "cache_key")


# ------------------------------------------------------------------
# (a) flag absent -> nothing changes
# ------------------------------------------------------------------

def test_int_index_returns_the_full_window_unchanged(dataset):
    out = dataset[0]

    assert "frame_index" not in out, "fixed-length path must not emit frame_index"
    assert set(out) == set(EXPECTED_FLOAT_FRAME_KEYS + EXPECTED_BOOL_FRAME_KEYS
                           + EXPECTED_CLIP_KEYS), (
        "the test's key inventory drifted from what __getitem__ emits; update "
        "PER_FRAME_CLIP_KEYS / NON_FRAME_CLIP_KEYS too if a new key is per-frame")

    full = list(range(NUM_FRAMES))
    for k in EXPECTED_FLOAT_FRAME_KEYS:
        assert out[k].shape[0] == NUM_FRAMES, k
        assert _recover(out[k]) == full, k
    for k in EXPECTED_BOOL_FRAME_KEYS:
        assert out[k].shape[0] == NUM_FRAMES, k
        assert out[k].reshape(NUM_FRAMES, -1)[:, 0].tolist() == [
            i % 2 == 0 for i in full], k

    assert torch.equal(out["cam_intrinsics"], torch.tensor([215.5, 112.0, 112.0]))
    assert out["cache_key"] == "seq_A_7"
    assert out["has_mano"] is True


# ------------------------------------------------------------------
# (b) flag set -> one index vector for every per-frame tensor
# ------------------------------------------------------------------

@pytest.mark.parametrize("n,seed", [(2, 1), (5, 12345), (9, 7), (16, 99)])
def test_tuple_index_slices_every_per_frame_tensor_with_the_same_indices(dataset, n, seed):
    out = dataset[(0, n, seed)]

    fi = out["frame_index"]
    assert fi.dtype == torch.long
    assert fi.tolist() == sorted(fi.tolist()), "sampled indices must stay in temporal order"
    assert len(set(fi.tolist())) == n, "sampled indices must be distinct"
    assert fi.min() >= 0 and fi.max() < NUM_FRAMES

    want = fi.tolist()
    recovered = {}
    for k in EXPECTED_FLOAT_FRAME_KEYS:
        assert out[k].shape[0] == n, f"{k} kept the wrong length"
        recovered[k] = _recover(out[k])
    assert all(v == want for v in recovered.values()), (
        f"per-frame tensors disagree about which frames were kept: {recovered}")

    for k in EXPECTED_BOOL_FRAME_KEYS:
        assert out[k].shape[0] == n, f"{k} kept the wrong length"
        assert out[k].reshape(n, -1)[:, 0].tolist() == [i % 2 == 0 for i in want], k

    # Per-clip keys must survive untouched.
    assert torch.equal(out["cam_intrinsics"], torch.tensor([215.5, 112.0, 112.0]))
    assert out["cache_key"] == "seq_A_7"


def test_same_seed_reproduces_the_same_subset(dataset):
    a = dataset[(0, 6, 4242)]
    b = dataset[(0, 6, 4242)]
    c = dataset[(0, 6, 4243)]
    assert torch.equal(a["frame_index"], b["frame_index"])
    assert not torch.equal(a["frame_index"], c["frame_index"])


def test_unregistered_per_frame_key_fails_loudly(dataset, monkeypatch):
    # Simulate "someone added a per-frame key and forgot to register it" by hiding one of the
    # real ones from the slice list. It must raise, not silently ship a 16-frame tensor into
    # a 5-frame batch.
    monkeypatch.setattr(
        T, "PER_FRAME_CLIP_KEYS",
        tuple(k for k in T.PER_FRAME_CLIP_KEYS if k != "da3_wrist"), raising=True)
    with pytest.raises(RuntimeError, match="da3_wrist"):
        dataset[(0, 5, 11)]


def test_mixed_dataset_forwards_the_tuple_payload(dataset, monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="random_frames_test_")
    other = _make_dataset(tmpdir)
    mixed = T.MixedHandDataset([dataset, other], ["a", "b"])

    assert len(mixed) == 2
    direct = other[(0, 6, 555)]
    through = mixed[(1, 6, 555)]          # global index 1 -> part "b", local clip 0
    assert torch.equal(direct["frame_index"], through["frame_index"])
    assert through["img"].shape[0] == 6
    # A bare int must still return the full window through the mix.
    assert "frame_index" not in mixed[1]
    assert mixed[1]["img"].shape[0] == NUM_FRAMES


# ------------------------------------------------------------------
# (c) n bounds, shared per batch, deterministic
# ------------------------------------------------------------------

def _batch_sampler(lo=2, hi=16, batch_size=4, n_items=20, seed=0):
    base = BatchSampler(SequentialSampler(list(range(n_items))), batch_size, drop_last=True)
    return T.RandomFrameCountBatchSampler(base, lo, hi, seed=seed)


def test_batch_sampler_shares_one_n_within_each_batch_and_respects_bounds():
    sampler = _batch_sampler()
    assert len(sampler) == 5, "len must match the wrapped BatchSampler so steps_per_epoch holds"

    seen = []
    for _ in range(100):                       # 100 epochs = 500 batches
        for batch in sampler:
            assert len(batch) == 4
            counts = {n for _, n, _ in batch}
            assert len(counts) == 1, f"batch mixes frame counts {counts}, collate would raise"
            n = counts.pop()
            assert 2 <= n <= 16, n
            seen.append(n)
            assert len({s for _, _, s in batch}) == 4, "per-sample subset seeds must differ"
    assert min(seen) == 2 and max(seen) == 16, (
        f"bounds should be inclusive, observed [{min(seen)}, {max(seen)}]")


def test_batch_sampler_is_deterministic_under_a_seed_and_varies_across_epochs():
    a, b = _batch_sampler(seed=7), _batch_sampler(seed=7)
    epochs_a = [list(a) for _ in range(3)]
    epochs_b = [list(b) for _ in range(3)]
    assert epochs_a == epochs_b, "same seed must reproduce the whole stream, epoch by epoch"

    different_seed = [list(_batch_sampler(seed=8)) for _ in range(3)]
    assert different_seed != epochs_a

    # The RNG is deliberately not reset per epoch: epoch 2 must not replay epoch 1.
    seeds = [{s for batch in epoch for _, _, s in batch} for epoch in epochs_a]
    assert seeds[0].isdisjoint(seeds[1]) and seeds[1].isdisjoint(seeds[2])


def test_shared_n_lets_the_real_collate_stack_the_batch(dataset):
    batch_indices = next(iter(_batch_sampler(lo=3, hi=12, batch_size=3, n_items=3, seed=5)))
    n = batch_indices[0][1]
    samples = [dataset[(0, n, seed)] for _, _, seed in batch_indices]
    collated = T.mixed_collate(samples)

    assert collated["img"].shape[:2] == (3, n)
    assert collated["cached_tokens"].shape == (3, n, TOKEN_P, TOKEN_C)
    assert collated["frame_index"].shape == (3, n)
    # Different subsets per sample is the Fast3R behaviour we want, and it survives collate.
    assert collated["frame_index"].unique(dim=0).shape[0] > 1


def test_dataloader_end_to_end_yields_uniform_length_batches(dataset):
    """The production wiring: DataLoader(batch_sampler=...) -> tuple index -> mixed_collate."""
    dataset.clips = dataset.clips * 4
    base = BatchSampler(SequentialSampler(range(4)), 2, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_sampler=T.RandomFrameCountBatchSampler(base, 2, 16, seed=3),
        num_workers=0, collate_fn=T.mixed_collate)

    assert len(loader) == 2, "steps_per_epoch must be unaffected by variable-length mode"
    for batch in loader:
        n = batch["img"].shape[1]
        assert 2 <= n <= 16
        for key in ("gt", "gt_joints", "hand_bboxes", "hand_valid", "cached_tokens",
                    "gt_joints_2d", "cam_extrinsics", "contact", "da3_wrist",
                    "gt_obj_depth", "gt_obj_mask", "frame_index"):
            assert batch[key].shape[1] == n, f"{key} came out of collate at the wrong length"
        assert batch["cam_intrinsics"].shape == (2, 3)   # per clip, never sliced
        # And the head-facing views must be built at that same n.
        views = T.build_views(batch["img"], batch["img"].shape[1], "cpu",
                              frame_index=batch["frame_index"])
        assert torch.equal(views["timestamp"], batch["frame_index"])


# ------------------------------------------------------------------
# (d) timestamps carry the true indices
# ------------------------------------------------------------------

def test_build_views_timestamps_are_the_true_sampled_indices():
    n = 5
    imgs = torch.zeros(2, n, 3, 8, 8)
    frame_index = torch.tensor([[0, 3, 7, 11, 15], [1, 2, 4, 9, 14]])
    views = T.build_views(imgs, n, "cpu", frame_index=frame_index)

    assert views["timestamp"].dtype == torch.long
    assert torch.equal(views["timestamp"], frame_index), (
        "timestamps must carry the real window positions, not 0..n-1, or the spacing that "
        "makes subsampling read as masked-out frames is erased")
    for k in ("is_target", "is_static", "valid_mask", "camera_poses", "camera_intrs", "depthmap"):
        assert views[k].shape[1] == n, f"{k} was built at the wrong length"


def test_build_views_without_frame_index_is_the_old_arange():
    imgs = torch.zeros(2, 16, 3, 8, 8)
    views = T.build_views(imgs, 16, "cpu")
    assert torch.equal(views["timestamp"],
                       torch.arange(16).unsqueeze(0).expand(2, -1))


def test_build_views_rejects_a_length_mismatch():
    imgs = torch.zeros(2, 5, 3, 8, 8)
    with pytest.raises(RuntimeError, match="imgs carries S=5"):
        T.build_views(imgs, 16, "cpu")


# ------------------------------------------------------------------
# Config validation
# ------------------------------------------------------------------

def test_parse_random_frames_accepts_a_valid_range():
    assert T.parse_random_frames([2, 32], 32) == (2, 32)
    assert T.parse_random_frames((4, 16), 16) == (4, 16)


def test_parse_random_frames_absent_key_is_none():
    assert T.parse_random_frames(None, 16) is None


@pytest.mark.parametrize("spec", [
    16,                 # not a list
    [8],                # one element
    [2, 8, 16],         # three elements
    [1, 16],            # min < 2
    [8, 8],             # min == max
    [12, 8],            # min > max
    [2, 32],            # max > num_frames
    ["a", "b"],         # not integers
])
def test_parse_random_frames_rejects_bad_specs(spec):
    with pytest.raises(SystemExit):
        T.parse_random_frames(spec, 16)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

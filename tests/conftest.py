"""CPU-test bootstrap for the hand-scene metric-coupling unit tests.

These tests cover three dependency-light, ``torch``-only modules:
  * the shared geometry helper ``...models.utils.hand_depth_sampling``
  * the L1 loss ``scripts.hand_depth_anchor_loss``
  * the L2 head  ``...models.heads.metric_scale_head``

In the full training environment importing them via their real package paths
works directly. In a bare CPU sandbox, ``import diffsynth`` eagerly runs
``diffsynth/__init__.py``, which pulls in heavy optional deps (e.g. modelscope)
that are not installed. To keep the tests runnable in BOTH places we try the
real import first and only fall back to lightweight namespace shims if it fails.
The shims point each parent package's ``__path__`` at the real source dir, so the
leaf modules import normally while ``diffsynth/__init__.py`` is never executed.

On the cluster (real deps present) this conftest is a no-op.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_HELPER_FQ = (
    "diffsynth.auxiliary_models.worldmirror.models.utils.hand_depth_sampling"
)

# Parent packages to shim (with __path__ at the real source dirs) when the heavy
# top-level import is unavailable. Order matters: parents before children.
_SHIM_PKGS = [
    "diffsynth",
    "diffsynth.auxiliary_models",
    "diffsynth.auxiliary_models.worldmirror",
    "diffsynth.auxiliary_models.worldmirror.models",
    "diffsynth.auxiliary_models.worldmirror.models.utils",
    "diffsynth.auxiliary_models.worldmirror.models.heads",
]


def _real_import_works() -> bool:
    try:
        importlib.import_module(_HELPER_FQ)
        return True
    except Exception:
        return False


def _install_shims() -> None:
    for name in _SHIM_PKGS:
        if name in sys.modules:
            continue
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(_ROOT.joinpath(*name.split(".")))]
        pkg.__package__ = name
        sys.modules[name] = pkg


if not _real_import_works():
    _install_shims()

from typing_extensions import Literal, TypeAlias

from ..auxiliary_models import WorldMirror

# Single-file loader detection table.
# Format: (state_dict_keys_hash, state_dict_keys_hash_with_shape, model_names, model_classes, model_resource)
# Only the WorldMirror reconstructor remains after the NeoVerse diffusion zoo was removed.
model_loader_configs = [
    (None, "1a1d001a35f78f3a7796a1e719ead340", ["reconstructor"], [WorldMirror], "civitai"),
]

# HuggingFace-format and patch loaders are unused by the reconstructor path.
huggingface_model_loader_configs = []
patch_model_loader_configs = []

# Preset download tables (consumed by downloader.py). Emptied: checkpoints are
# loaded by explicit --reconstructor_path, not by preset id.
preset_models_on_huggingface = {}
preset_models_on_modelscope = {}

Preset_model_id: TypeAlias = Literal["reconstructor"]

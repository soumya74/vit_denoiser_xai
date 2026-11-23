import os
import sys
import inspect
import zipfile
import shutil
import urllib.request
import numpy as np
import torch

MAE_CONFIGS = {
    "base": {
        "arch": "mae_vit_base_patch16",
        "filename": "mae_pretrain_vit_base.pth",
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        "patch_size": 16,
        "num_layers": 12,
        "embed_dim": 768,
    },
    "large": {
        "arch": "mae_vit_large_patch16",
        "filename": "mae_pretrain_vit_large.pth",
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
        "patch_size": 16,
        "num_layers": 24,
        "embed_dim": 1024,
    },
    "huge": {
        "arch": "mae_vit_huge_patch14",
        "filename": "mae_pretrain_vit_huge.pth",
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
        "patch_size": 14,
        "num_layers": 32,
        "embed_dim": 1280,
    },
}


def ensure_mae_repo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mae_path = os.path.join(base_dir, "mae")

    if not os.path.exists(mae_path):
        zip_url = "https://github.com/facebookresearch/mae/archive/refs/heads/main.zip"
        zip_path = os.path.join(base_dir, "mae_repo.zip")
        extracted_dir = os.path.join(base_dir, "mae-main")

        print("MAE repository not found. Downloading ZIP from GitHub...")
        urllib.request.urlretrieve(zip_url, zip_path)

        print("Extracting MAE repository...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base_dir)

        shutil.move(extracted_dir, mae_path)
        os.remove(zip_path)
        print(f"MAE repository ready at: {mae_path}")

    if mae_path not in sys.path:
        sys.path.insert(0, mae_path)
    return mae_path


def ensure_mae_weights(variant="base", weights_path=None):
    if variant not in MAE_CONFIGS:
        raise ValueError(f"Unsupported MAE variant '{variant}'. Choose from: {list(MAE_CONFIGS.keys())}")

    cfg = MAE_CONFIGS[variant]
    if weights_path is None:
        model_dir = os.path.abspath("model_zoo")
        os.makedirs(model_dir, exist_ok=True)
        weights_path = os.path.join(model_dir, cfg["filename"])

    if not os.path.exists(weights_path):
        print(f"Downloading MAE-{variant} pre-trained weights to {weights_path}...")
        urllib.request.urlretrieve(cfg["url"], weights_path)
        print("Download complete.")

    return weights_path


def _patch_timm_block_compat(models_mae):
    """Handles newer timm versions where `qk_scale` was removed from VisionTransformer Block."""
    if "qk_scale" not in inspect.signature(models_mae.Block.__init__).parameters:
        orig_block = models_mae.Block

        class CompatBlock(orig_block):
            def __init__(self, *args, qk_scale=None, **kwargs):
                super().__init__(*args, **kwargs)

        models_mae.Block = CompatBlock


def load_mae_model(variant="base", weights_path=None, target_device="cpu"):
    ensure_mae_repo()
    import models_mae

    _patch_timm_block_compat(models_mae)

    # Compatibility shim for NumPy 1.24+ where deprecated `np.float` was removed
    if not hasattr(np, "float"):
        np.float = float

    cfg = MAE_CONFIGS[variant]
    resolved_weights = ensure_mae_weights(variant, weights_path)

    model = getattr(models_mae, cfg["arch"])()
    checkpoint = torch.load(resolved_weights, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Official pretrain checkpoints contain encoder weights only; strict=False is expected
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded MAE-{variant} ({cfg['arch']}) | Missing keys: {len(msg.missing_keys)} (expected for encoder-only pretrain)")

    model.eval()
    return model.to(target_device), cfg
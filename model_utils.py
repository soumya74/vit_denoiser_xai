import os
import sys
import torch

def load_swinir_model(weights_path, target_device="cpu"):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    candidate_paths = [
        os.path.join(current_dir, "SwinIR"),
        os.path.abspath("SwinIR"),
        "/content/vit_denoiser_xai/SwinIR"
    ]

    swinir_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            swinir_path = path
            break

    if swinir_path and swinir_path not in sys.path:
        sys.path.append(swinir_path)

    try:
        from models.network_swinir import SwinIR
    except ModuleNotFoundError:
        sys.path.append(os.getcwd())
        try:
            from SwinIR.models.network_swinir import SwinIR
        except ModuleNotFoundError:
            raise RuntimeError("SwinIR module not found. Clone repository into directory.")

    model = SwinIR(
        upscale=1, in_chans=3, img_size=128, window_size=8,
        img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
        upsampler="", resi_connection="1conv"
    )

    if not os.path.isabs(weights_path):
        weights_path = os.path.join(current_dir, weights_path)

    if not os.path.exists(weights_path):
        if os.path.exists(os.path.abspath(weights_path)):
            weights_path = os.path.abspath(weights_path)
        else:
            raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    pretrained_model = torch.load(weights_path, map_location="cpu")
    param_key_g = "params"

    if param_key_g in pretrained_model:
        model.load_state_dict(pretrained_model[param_key_g], strict=True)
    else:
        model.load_state_dict(pretrained_model, strict=True)

    model.eval()
    return model.to(target_device)
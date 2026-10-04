import torch
import sys
import os

# Importing IPEX registers the "xpu" device in standard PyTorch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    print("Warning: intel-extension-for-pytorch not found. Falling back to CPU.")

def load_swinir_model(weights_path, target_device="cpu"):
    """Initializes the SwinIR Denoising model and moves it to the target device."""
    if not os.path.exists("SwinIR"):
        raise RuntimeError("SwinIR folder not found. Run setup_env.py first.")
        
    sys.path.append('SwinIR')
    from models.network_swinir import SwinIR

    # SwinIR parameters for Color Denoising
    model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                   num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='', resi_connection='1conv')
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['params'], strict=True)
    model.eval()

    # Move the model to the target device (CPU, CUDA, or XPU)
    model = model.to(target_device)
    
    # Optional but recommended for Intel hardware: 
    # ipex.optimize applies hardware-specific optimizations to the PyTorch model
    if target_device == "xpu":
        model = ipex.optimize(model)

    return model
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from setup_env import setup_project
from dataset import load_image, add_gaussian_noise
from model_utils import load_swinir_model
from analyzer import ViTXAIAnalyzer
from visualizer import plot_xai_results

# Import IPEX to enable the xpu device checks
try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except ImportError:
    has_ipex = False

def main():
    setup_project()
    
    # Standard PyTorch device selection logic
    if has_ipex and hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device('xpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    device = 'cpu'
        
    print(f"Using device: {device}")
    
    image_path = "0002.png"
    if not os.path.exists(image_path):
        print(f"ERROR: '{image_path}' not found.")
        return
        
    print("Preparing images...")
    clean_tensor, clean_img_np = load_image(image_path)
    noisy_tensor = add_gaussian_noise(clean_tensor, noise_level=50)
    
    # Send tensors to the XPU/NPU natively
    clean_tensor = clean_tensor.to(device)
    noisy_tensor = noisy_tensor.to(device)
    
    print("Loading SwinIR model...")
    weights_path = "model_zoo/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
    model = load_swinir_model(weights_path, target_device=device)
    
    print("Running XAI inference and calculating drift...")
    analyzer = ViTXAIAnalyzer(model)
    spatial_heatmap, channel_mse, top_channels, noise_mask = analyzer.analyze_drift(clean_tensor, noisy_tensor)

    # Convert from PyTorch [1, 3, H, W] to NumPy [H, W, 3]
    mask_visual = noise_mask.squeeze().numpy().transpose(1, 2, 0)
    
    # The mask contains negative values predicting the inverse of the noise (-N).
    # Adding 0.5 shifts the values so that 0 (no noise) renders as a neutral gray background.
    mask_visual = np.clip(mask_visual + 0.5, 0, 1)
    
    plt.imsave("predicted_noise_mask.png", mask_visual)
    print("Noise mask PNG dumped to 'predicted_noise_mask.png'")
    # --------------------------------------------------    
    print("Generating visualizations...")
    plot_xai_results(clean_img_np, noisy_tensor, spatial_heatmap, channel_mse, top_channels,output_file="xai_results_0002.png")
    
    print("Done!")

if __name__ == "__main__":
    main()
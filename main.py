import os
import sys
import glob
import argparse
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt

# python main.py --num_images 70 --image_dir ..\CBSD68-dataset\noisy5\train\clean
# Colab !python main.py --num_images 70

def ensure_swinir():
    swinir_path = os.path.abspath("SwinIR")
    if not os.path.exists(swinir_path):
        print("SwinIR not found. Cloning repository...")
        subprocess.run(["git", "clone", "https://github.com/JingyunLiang/SwinIR.git", swinir_path], check=True)
    if swinir_path not in sys.path:
        sys.path.append(swinir_path)

ensure_swinir()
from model_utils import load_swinir_model
from analyzer import ViTXAIAnalyzer
from dataset import load_image, add_gaussian_noise

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Stripped out all Intel/XPU logic to prevent DLL crashes
    return torch.device("cpu")

def process_images(args):
    device = get_device()
    print(f"Executing on hardware backend: {device}")

    model = load_swinir_model(args.weights_path, target_device=device)
    analyzer = ViTXAIAnalyzer(model)

    image_paths = glob.glob(os.path.join(args.image_dir, "*.[jp][pn]g"))[:args.num_images]
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    all_channel_mse = []

    print(f"Processing {len(image_paths)} images...")
    for idx, img_path in enumerate(image_paths):
        print(f"Processing {idx + 1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        clean_tensor, _ = load_image(img_path)
        noisy_tensor = add_gaussian_noise(clean_tensor, noise_level=75)
        
        clean_tensor = clean_tensor.to(device)
        noisy_tensor = noisy_tensor.to(device)
        
        _, channel_mse, _, _ = analyzer.analyze_drift(clean_tensor, noisy_tensor)
        all_channel_mse.append(channel_mse)

    if not all_channel_mse:
        return

    all_channel_mse = np.array(all_channel_mse)
    print("\nFinished processing. Shape:", all_channel_mse.shape)

    # 1. Average channel corruption
    mean_channel_mse = np.mean(all_channel_mse, axis=0)
    top20 = np.argsort(mean_channel_mse)[::-1][:20]
    
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(top20)), mean_channel_mse[top20])
    plt.xticks(np.arange(len(top20)), top20, rotation=45)
    plt.title("Top 20 Most Noise-Sensitive Channels")
    plt.xlabel("Channel Index")
    plt.ylabel("Average MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "mean_channel_corruption.png"))
    plt.close()

    # 2. Top-10 frequency
    channel_frequency = np.zeros(all_channel_mse.shape[1])
    for mse in all_channel_mse:
        for ch in np.argsort(mse)[::-1][:10]:
            channel_frequency[ch] += 1
            
    plt.figure(figsize=(15, 5))
    plt.bar(np.arange(len(channel_frequency)), channel_frequency)
    plt.title("Frequency of Channel Appearing in Top-10")
    plt.xlabel("Channel Index")
    plt.ylabel(f"Count (out of {len(image_paths)} images)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "channel_frequency.png"))
    plt.close()

    # 3. Image x Channel heatmap
    plt.figure(figsize=(16, 8))
    plt.imshow(all_channel_mse, cmap="magma", aspect="auto")
    plt.colorbar(label="MSE")
    plt.xlabel("Channel Index")
    plt.ylabel("Image Number")
    plt.title("Channel Corruption Across Images")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "channel_heatmap.png"))
    plt.close()

    # Print dominant channels
    dominant_channels = np.argsort(channel_frequency)[::-1][:10]
    print("\nTop Consistently Corrupted Channels\n")
    for ch in dominant_channels:
        print(f"Channel: {ch} Frequency: {int(channel_frequency[ch])} Mean MSE: {mean_channel_mse[ch]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="./test_images")
    parser.add_argument("--output_dir", type=str, default="./xai_outputs")
    parser.add_argument("--weights_path", type=str, default="./model_zoo/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth")
    parser.add_argument("--num_images", type=int, default=50)
    args = parser.parse_args()
    process_images(args)
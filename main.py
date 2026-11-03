import os
import sys
import glob
import argparse
import subprocess
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

def ensure_swinir():
    swinir_path = os.path.abspath("SwinIR")
    if not os.path.exists(swinir_path):
        subprocess.run(["git", "clone", "https://github.com/JingyunLiang/SwinIR.git", swinir_path], check=True)
    if swinir_path not in sys.path:
        sys.path.append(swinir_path)

ensure_swinir()
from model_utils import load_swinir_model
from dataset import load_image, add_gaussian_noise
from analyzer import ViTXAIAnalyzer

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def save_image(tensor, path, shift_center=False):
    img = tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    if shift_center:
        img = np.clip(img + 0.5, 0, 1) 
    else:
        img = np.clip(img, 0, 1)
    plt.imsave(path, img)

class LatentSpaceMasker:
    def __init__(self, model, top_k=15):
        self.model = model
        self.top_k = top_k
        self.clean_features = None
        self.hook_clean = None
        self.hook_mask = None

    def _clean_hook(self, module, args):
        self.clean_features = args[0].detach()

    def _mask_hook(self, module, args):
        noisy_features = args[0]
        mse_per_channel = torch.mean((self.clean_features - noisy_features)**2, dim=[0, 2, 3])
        top_k_indices = torch.argsort(mse_per_channel, descending=True)[:self.top_k]
        
        mask = torch.zeros_like(noisy_features)
        mask[:, top_k_indices, :, :] = 1.0
        
        return (noisy_features * mask, )

    def extract_clean_reference(self, clean_tensor):
        self.hook_clean = self.model.conv_after_body.register_forward_pre_hook(self._clean_hook)
        with torch.no_grad():
            _ = self.model(clean_tensor)
        self.hook_clean.remove()

    def forward_with_truncated_dimensions(self, noisy_tensor):
        self.hook_mask = self.model.conv_after_body.register_forward_pre_hook(self._mask_hook)
        with torch.no_grad():
            output = self.model(noisy_tensor)
        self.hook_mask.remove()
        return output

def run_experiment(args):
    device = get_device()
    print(f"Running experiment on: {device}")

    model = load_swinir_model(args.weights_path, target_device=device)
    analyzer = ViTXAIAnalyzer(model)
    masker = LatentSpaceMasker(model, top_k=args.top_k)

    image_paths = glob.glob(os.path.join(args.image_dir, "*.[jp][pn]g"))[:args.num_images]
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = open(os.path.join(args.output_dir, "experiment_metrics.csv"), "w")
    csv_file.write("Image,PSNR_Clean_vs_Standard,PSNR_Clean_vs_Top15,PSNR_Standard_vs_Top15\n")

    all_channel_mse = []

    for idx, img_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx+1}/{len(image_paths)}] Processing {base_name}...")

        clean_tensor, _ = load_image(img_path)
        noisy_tensor = add_gaussian_noise(clean_tensor, noise_level=args.noise_level)
        
        clean_tensor = clean_tensor.to(device)
        noisy_tensor = noisy_tensor.to(device)

        # 1. Standard Forward Pass
        with torch.no_grad():
            out_standard = model(noisy_tensor)
        noise_dump_standard = noisy_tensor - out_standard

        # 1b. XAI Analysis (for charts)
        _, channel_mse, _, _ = analyzer.analyze_drift(clean_tensor, noisy_tensor)
        all_channel_mse.append(channel_mse)

        # 2. Truncated Forward Pass
        masker.extract_clean_reference(clean_tensor)
        out_truncated = masker.forward_with_truncated_dimensions(noisy_tensor)
        noise_dump_truncated = noisy_tensor - out_truncated

        # 3. Calculate Metrics
        psnr_std = calculate_psnr(clean_tensor, out_standard)
        psnr_trunc = calculate_psnr(clean_tensor, out_truncated)
        psnr_diff = calculate_psnr(out_standard, out_truncated)
        csv_file.write(f"{base_name},{psnr_std:.2f},{psnr_trunc:.2f},{psnr_diff:.2f}\n")

        # 4. Save Image Artifacts
        save_image(out_standard, os.path.join(args.output_dir, f"{base_name}_01_denoised_standard.png"))
        save_image(out_truncated, os.path.join(args.output_dir, f"{base_name}_02_denoised_top15.png"))
        save_image(noise_dump_standard, os.path.join(args.output_dir, f"{base_name}_03_dump_standard.png"), shift_center=True)
        save_image(noise_dump_truncated, os.path.join(args.output_dir, f"{base_name}_04_dump_top15.png"), shift_center=True)

    csv_file.close()

    # 5. Generate Global XAI Plots
    if all_channel_mse:
        all_channel_mse = np.array(all_channel_mse)
        
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

        plt.figure(figsize=(16, 8))
        plt.imshow(all_channel_mse, cmap="magma", aspect="auto")
        plt.colorbar(label="MSE")
        plt.xlabel("Channel Index")
        plt.ylabel("Image Number")
        plt.title("Channel Corruption Across Images")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "channel_heatmap.png"))
        plt.close()

    print(f"\nExperiment complete. Data, images, and XAI charts saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="./test_images")
    parser.add_argument("--output_dir", type=str, default="./truncation_experiment")
    parser.add_argument("--weights_path", type=str, default="./model_zoo/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth")
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--noise_level", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=15)
    
    args = parser.parse_args()
    run_experiment(args)
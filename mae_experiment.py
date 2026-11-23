import os
import glob
import numpy as np
import torch

from dataset import load_image_for_mae, add_gaussian_noise, normalize_imagenet
from mae_utils import load_mae_model
from mae_analyzer import MAEXAIAnalyzer
from visualizer import plot_mae_layer_comparison, plot_mae_global_summary


def run_mae_experiment(args, device):
    print(f"Running MAE ({args.mae_variant}) embedding drift experiment on: {device}")

    model, cfg = load_mae_model(
        variant=args.mae_variant,
        weights_path=args.mae_weights_path,
        target_device=device,
    )
    analyzer = MAEXAIAnalyzer(
        model=model,
        patch_size=cfg["patch_size"],
        img_size=224,
        layer_step=args.layer_step,
    )
    target_layers = analyzer.target_layers
    print(f"Tracking encoder layers: {target_layers}")

    image_paths = glob.glob(os.path.join(args.image_dir, "*.[jp][pn]g"))[:args.num_images]
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return

    out_dir = os.path.abspath(args.output_dir)
    per_img_dir = os.path.join(out_dir, "per_image_visuals")
    os.makedirs(per_img_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "mae_layer_metrics.csv")
    csv_file = open(csv_path, "w")
    csv_file.write("Image,Layer,Mean_Patch_CosDist,CLS_CosDist,Mean_L2_Dist,Max_Channel_MSE\n")

    # Accumulators per layer across the dataset
    layer_channel_mse = {l: [] for l in target_layers}
    layer_cos_dist = {l: [] for l in target_layers}
    layer_l2_dist = {l: [] for l in target_layers}

    for idx, img_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx + 1}/{len(image_paths)}] Processing {base_name}...")

        # 1. Load resized (224x224) image in [0, 1], add noise in pixel space, then normalize
        clean_01, clean_np = load_image_for_mae(img_path, img_size=224)
        noisy_01 = add_gaussian_noise(clean_01, noise_level=args.noise_level)

        clean_norm = normalize_imagenet(clean_01).to(device)
        noisy_norm = normalize_imagenet(noisy_01).to(device)

        # 2. Extract multi-layer embedding drift
        layer_results = analyzer.analyze_drift(clean_norm, noisy_norm)

        # 3. Log metrics
        for l in target_layers:
            res = layer_results[l]
            layer_channel_mse[l].append(res["channel_mse"])
            layer_cos_dist[l].append(res["mean_cos_dist"])
            layer_l2_dist[l].append(res["mean_l2_dist"])
            csv_file.write(
                f"{base_name},{l},{res['mean_cos_dist']:.6f},"
                f"{res['cls_cos_dist']:.6f},{res['mean_l2_dist']:.6f},"
                f"{float(np.max(res['channel_mse'])):.6f}\n"
            )

        # 4. Save per-image multi-layer visual comparison (capped by --max_visuals)
        if idx < args.max_visuals:
            vis_path = os.path.join(per_img_dir, f"{base_name}_mae_{args.mae_variant}_layers.png")
            plot_mae_layer_comparison(clean_np, noisy_01, layer_results, vis_path)

    csv_file.close()
    analyzer.remove_hooks()

    # 5. Generate dataset-wide summary plots
    plot_mae_global_summary(
        layer_channel_mse=layer_channel_mse,
        layer_cos_dist=layer_cos_dist,
        layer_l2_dist=layer_l2_dist,
        output_dir=out_dir,
        variant=args.mae_variant,
    )

    print(f"\nMAE experiment complete. Metrics and plots saved to: {out_dir}")
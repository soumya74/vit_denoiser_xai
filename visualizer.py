import os
import matplotlib.pyplot as plt
import numpy as np


def plot_xai_results(clean_img_np, noisy_tensor, spatial_heatmap, channel_mse, top_channels, output_file="xai_results.png"):
    """Generates a 2x2 grid showing the images and the XAI analysis results (SwinIR)."""
    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(clean_img_np)
    ax1.set_title("Clean Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    noisy_img_np = noisy_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    ax2.imshow(np.clip(noisy_img_np, 0, 1))
    ax2.set_title("Input Noisy Image")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(spatial_heatmap, cmap="magma")
    ax3.set_title("Encoder Output Drift (Spatial Patch Corruption)")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(range(10), channel_mse[top_channels], color="crimson")
    ax4.set_xticks(range(10))
    ax4.set_xticklabels(top_channels)
    ax4.set_title("Top 10 Most Corrupted Latent Channels")
    ax4.set_xlabel("Channel Index")
    ax4.set_ylabel("Mean Squared Error")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Visualization saved to '{output_file}'.")


def plot_mae_layer_comparison(clean_img_np, noisy_tensor_01, layer_results, output_file):
    """
    Plots clean/noisy inputs alongside spatial patch drift heatmaps and
    top-10 corrupted embedding dimensions for every tracked MAE layer.
    """
    layers = sorted(layer_results.keys())
    num_layers = len(layers)
    noisy_np = np.clip(noisy_tensor_01.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1)

    fig, axes = plt.subplots(2, num_layers + 1, figsize=(4.2 * (num_layers + 1), 8))

    # Column 0: Clean & Noisy reference images
    axes[0, 0].imshow(clean_img_np)
    axes[0, 0].set_title("Clean (224x224)")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(noisy_np)
    axes[1, 0].set_title("Noisy Input")
    axes[1, 0].axis("off")

    # Columns 1..N: Spatial Cosine Drift (row 0) & Top-10 Channel MSE (row 1)
    for col_idx, l in enumerate(layers, start=1):
        res = layer_results[l]

        im = axes[0, col_idx].imshow(res["spatial_heatmap"], cmap="magma")
        axes[0, col_idx].set_title(f"Layer {l} Spatial Drift\n(CosDist={res['mean_cos_dist']:.3f})")
        axes[0, col_idx].axis("off")
        fig.colorbar(im, ax=axes[0, col_idx], fraction=0.046, pad=0.04)

        top_ch = res["top_channels"]
        axes[1, col_idx].bar(range(len(top_ch)), res["channel_mse"][top_ch], color="crimson")
        axes[1, col_idx].set_xticks(range(len(top_ch)))
        axes[1, col_idx].set_xticklabels(top_ch, rotation=45, fontsize=8)
        axes[1, col_idx].set_title(f"Layer {l} Top-10 Channels")
        axes[1, col_idx].set_ylabel("MSE")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def plot_mae_global_summary(layer_channel_mse, layer_cos_dist, layer_l2_dist, output_dir, variant="base"):
    """Generates dataset-wide MAE drift progression and per-layer channel heatmaps."""
    layers = sorted(layer_channel_mse.keys())
    mean_cos = [np.mean(layer_cos_dist[l]) for l in layers]
    mean_l2 = [np.mean(layer_l2_dist[l]) for l in layers]

    # 1. Layer-wise Drift Progression (Cosine Distance & L2 Norm)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "#1f77b4"
    ax1.plot(layers, mean_cos, marker="o", color=color1, linewidth=2, label="Mean Patch Cosine Distance")
    ax1.set_xlabel("MAE Encoder Layer")
    ax1.set_ylabel("Cosine Distance", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(layers)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.plot(layers, mean_l2, marker="s", linestyle="--", color=color2, linewidth=2, label="Mean L2 Drift")
    ax2.set_ylabel("L2 Norm", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title(f"MAE-{variant.capitalize()} Embedding Drift Across Encoder Depth")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "00_mae_layer_drift_progression.png"), dpi=150)
    plt.close()

    # 2. Per-layer Top-20 Corrupted Channels & Image-by-Channel Heatmaps
    for l in layers:
        mat = np.array(layer_channel_mse[l])  # (num_images, D)
        mean_mse = np.mean(mat, axis=0)
        top20 = np.argsort(mean_mse)[::-1][:20]

        plt.figure(figsize=(12, 4.5))
        plt.bar(np.arange(len(top20)), mean_mse[top20])
        plt.xticks(np.arange(len(top20)), top20, rotation=45)
        plt.title(f"MAE-{variant.capitalize()} Layer {l}: Top 20 Most Noise-Sensitive Dimensions")
        plt.xlabel("Embedding Dimension Index")
        plt.ylabel("Average MSE")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"layer_{l:02d}_top20_channels.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(16, 6))
        plt.imshow(mat, cmap="magma", aspect="auto")
        plt.colorbar(label="MSE")
        plt.xlabel("Embedding Dimension Index")
        plt.ylabel("Image Number")
        plt.title(f"MAE-{variant.capitalize()} Layer {l}: Dimension Corruption Across Images")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"layer_{l:02d}_channel_heatmap.png"), dpi=150)
        plt.close()
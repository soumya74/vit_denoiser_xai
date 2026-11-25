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
    Plots clean/noisy inputs alongside spatial patch drift heatmaps and top-10
    corrupted embedding dimensions across every tracked MAE encoder layer.
    """
    layers = sorted(layer_results.keys())
    num_layers = len(layers)
    noisy_np = np.clip(noisy_tensor_01.squeeze().cpu().numpy().transpose(1, 2, 0), 0, 1)

    total_cols = num_layers + 1
    fig, axes = plt.subplots(2, total_cols, figsize=(4.2 * total_cols, 8))

    # Column 0: Clean & Noisy reference inputs
    axes[0, 0].imshow(clean_img_np)
    axes[0, 0].set_title("Clean Input (224x224)")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(noisy_np)
    axes[1, 0].set_title("Noisy Input")
    axes[1, 0].axis("off")

    # Columns 1..N: Spatial Cosine Drift (row 0) & Top-10 Channel NSR (row 1)
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
        axes[1, col_idx].set_ylabel("Relative MSE (NSR)")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def plot_mae_global_summary(layer_channel_mse, layer_cos_dist, layer_l2_dist, output_dir, variant="base"):
    """Generates dataset-wide MAE drift progression, per-layer bar/heatmaps, and cross-layer per-channel dumps."""
    layers = sorted(layer_channel_mse.keys())
    mean_cos = [np.mean(layer_cos_dist[l]) for l in layers]
    mean_l2 = [np.mean(layer_l2_dist[l]) for l in layers]

    # 1. Global Layer-by-Layer Cosine & L2 Progression
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
    ax2.set_ylabel("L2 Norm (LayerNorm Calibrated)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title(f"MAE-{variant.capitalize()} Embedding Drift Across Encoder Depth")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "00_mae_layer_drift_progression.png"), dpi=150)
    plt.close()

    # 2. Cross-Layer Per-Dimension Trajectory Matrix (D x num_layers)
    dim_layer_matrix = np.column_stack([np.mean(np.array(layer_channel_mse[l]), axis=0) for l in layers])
    num_dims = dim_layer_matrix.shape[0]
    mean_across_layers = np.mean(dim_layer_matrix, axis=1)
    max_across_layers = np.max(dim_layer_matrix, axis=1)
    final_layer_nsr = dim_layer_matrix[:, -1]
    final_layer_share_pct = 100.0 * final_layer_nsr / (np.sum(final_layer_nsr) + 1e-8)

    # Sort dimensions by average NSR across all tracked encoder layers (descending)
    ranked_dims = np.argsort(mean_across_layers)[::-1]

    # Save complete 768-dimension cross-layer trajectory table
    traj_csv_path = os.path.join(output_dir, "channel_drift_across_layers.csv")
    with open(traj_csv_path, "w") as f:
        layer_headers = ",".join([f"Layer_{l:02d}_NSR" for l in layers])
        f.write(f"Rank,Dimension_Idx,Mean_NSR,Max_NSR,Final_Layer_Share_Pct,{layer_headers}\n")
        for rank, d in enumerate(ranked_dims, start=1):
            vals = ",".join([f"{dim_layer_matrix[d, j]:.6f}" for j in range(len(layers))])
            f.write(
                f"{rank},{d},{mean_across_layers[d]:.6f},{max_across_layers[d]:.6f},"
                f"{final_layer_share_pct[d]:.4f},{vals}\n"
            )

    # Plot cross-layer per-channel trajectories (Top-10 lines + All-768 sorted heatmap)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

    top10_dims = ranked_dims[:10]
    for d in top10_dims:
        ax_left.plot(
            layers,
            dim_layer_matrix[d],
            marker="o",
            linewidth=2,
            label=f"Dim {d} (mean={mean_across_layers[d]:.2f})",
        )
    ax_left.set_title(f"MAE-{variant.capitalize()}: Top-10 Noise-Concentrating Dimensions Across Depth")
    ax_left.set_xlabel("MAE Encoder Layer")
    ax_left.set_ylabel("Dataset-Avg Relative MSE (NSR)")
    ax_left.set_xticks(layers)
    ax_left.grid(True, linestyle="--", alpha=0.5)
    ax_left.legend(loc="upper left", fontsize=8.5, ncol=2)

    sorted_matrix = dim_layer_matrix[ranked_dims]  # (D, num_layers)
    im = ax_right.imshow(sorted_matrix, cmap="magma", aspect="auto")
    ax_right.set_title(f"All {num_dims} Dimensions Sorted by Noise Concentration Across Depth")
    ax_right.set_xlabel("MAE Encoder Layer")
    ax_right.set_ylabel("Dimension Rank (0 = Highest Noise Concentration)")
    ax_right.set_xticks(range(len(layers)))
    ax_right.set_xticklabels(layers)
    fig.colorbar(im, ax=ax_right, label="Relative MSE (NSR)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_top_channels_layer_trajectories.png"), dpi=150)
    plt.close()

    # 3. Individual Per-Layer Top-20 Bar Charts & Image-vs-Dimension Heatmaps
    for l in layers:
        mat = np.array(layer_channel_mse[l])  # (num_images, D)
        mean_mse = np.mean(mat, axis=0)
        top20 = np.argsort(mean_mse)[::-1][:20]

        plt.figure(figsize=(12, 4.5))
        plt.bar(np.arange(len(top20)), mean_mse[top20])
        plt.xticks(np.arange(len(top20)), top20, rotation=45)
        plt.title(f"MAE-{variant.capitalize()} Layer {l}: Top 20 Most Noise-Sensitive Dimensions")
        plt.xlabel("Embedding Dimension Index")
        plt.ylabel("Average Relative MSE (NSR)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"layer_{l:02d}_top20_channels.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(16, 6))
        plt.imshow(mat, cmap="magma", aspect="auto")
        plt.colorbar(label="Relative MSE (NSR)")
        plt.xlabel("Embedding Dimension Index")
        plt.ylabel("Image Number")
        plt.title(f"MAE-{variant.capitalize()} Layer {l}: Dimension Corruption Across Images")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"layer_{l:02d}_channel_heatmap.png"), dpi=150)
        plt.close()
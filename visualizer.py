import matplotlib.pyplot as plt
import numpy as np

def plot_xai_results(clean_img_np, noisy_tensor, spatial_heatmap, channel_mse, top_channels, output_file="xai_results.png"):
    """Generates a 2x2 grid showing the images and the XAI analysis results."""
    fig = plt.figure(figsize=(16, 10))

    # Plot Clean Image
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(clean_img_np)
    ax1.set_title("Clean Image")
    ax1.axis('off')

    # Plot Noisy Image
    ax2 = fig.add_subplot(2, 2, 2)
    noisy_img_np = noisy_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    ax2.imshow(np.clip(noisy_img_np, 0, 1))
    ax2.set_title("Input Noisy Image")
    ax2.axis('off')

    # Plot Latent Spatial Heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(spatial_heatmap, cmap='magma')
    ax3.set_title("Encoder Output Drift (Spatial Patch Corruption)")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    # Plot Channel Corruption
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(range(10), channel_mse[top_channels], color='crimson')
    ax4.set_xticks(range(10))
    ax4.set_xticklabels(top_channels)
    ax4.set_title("Top 10 Most Corrupted Latent Channels")
    ax4.set_xlabel("Channel Index")
    ax4.set_ylabel("Mean Squared Error")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Visualization saved to '{output_file}'.")
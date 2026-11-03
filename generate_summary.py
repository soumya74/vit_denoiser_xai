import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def generate_summary():
    output_dir = "./truncation_experiment"
    csv_path = os.path.join(output_dir, "experiment_metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"Metrics file not found at {csv_path}. Please run main.py first.")
        return

    df = pd.read_csv(csv_path)

    # ====================================================
    # 1. Generate PSNR Trend Plot (All 50 Images)
    # ====================================================
    plt.figure(figsize=(14, 6))
    
    plt.plot(df.index, df['PSNR_Clean_vs_Standard'], label='Standard (180 Channels)', 
             marker='o', markersize=6, alpha=0.8, color='#1f77b4', linewidth=2)
    plt.plot(df.index, df['PSNR_Clean_vs_Top15'], label='Truncated (Top 15 Channels)', 
             marker='x', markersize=6, alpha=0.8, color='#ff7f0e', linewidth=2)
    
    plt.title('Denoising Performance Across Dataset: Full Model vs Top-15 Dimensions', fontsize=14, pad=15)
    plt.xlabel('Image Index', fontsize=12)
    plt.ylabel('PSNR (dB) - Higher is Better', fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    trend_path = os.path.join(output_dir, "00_summary_psnr_trend.png")
    plt.savefig(trend_path, dpi=150)
    plt.close()

    # ====================================================
    # 2. Generate Visual Comparison Grid (First 5 Images)
    # ====================================================
    images_to_show = min(5, len(df))
    # Create a grid: rows = number of images, cols = 4 (the 4 output types)
    fig, axes = plt.subplots(images_to_show, 4, figsize=(18, 4.5 * images_to_show))
    
    cols = ['Standard Denoised\n(180 Channels)', 'Truncated Denoised\n(Top 15 Channels)', 
            'Standard Noise Mask', 'Truncated Noise Mask']
    
    # Set headers
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=16, fontweight='bold', pad=15)
        
    for i in range(images_to_show):
        base_name = str(df.iloc[i]['Image'])
        
        # Load the 4 generated artifacts for this image
        files = [
            os.path.join(output_dir, f"{base_name}_01_denoised_standard.png"),
            os.path.join(output_dir, f"{base_name}_02_denoised_top15.png"),
            os.path.join(output_dir, f"{base_name}_03_dump_standard.png"),
            os.path.join(output_dir, f"{base_name}_04_dump_top15.png")
        ]
        
        for j, f in enumerate(files):
            if os.path.exists(f):
                img = mpimg.imread(f)
                axes[i, j].imshow(img)
            axes[i, j].axis('off')
            
    plt.tight_layout()
    grid_path = os.path.join(output_dir, "00_summary_visual_grid.png")
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print("\nPresentation summaries generated successfully!")
    print(f"1. PSNR Chart: {trend_path}")
    print(f"2. Image Grid: {grid_path}")

if __name__ == "__main__":
    generate_summary()
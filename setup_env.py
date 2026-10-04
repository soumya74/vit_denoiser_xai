import os
import subprocess
import urllib.request

def setup_project():
    """Clones the SwinIR repo and downloads pre-trained weights."""
    print("--- Starting Project Setup ---")
    
    # 1. Clone SwinIR if not exists
    if not os.path.exists("SwinIR"):
        print("Cloning SwinIR repository...")
        #subprocess.check_call(["git", "clone", "https://github.com/JingyunLiang/SwinIR.git"])
    else:
        print("SwinIR repository already exists.")

    # 2. Download pre-trained weights
    model_dir = "model_zoo"
    os.makedirs(model_dir, exist_ok=True)
    weights_path = os.path.join(model_dir, "005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth")
    weights_url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
    
    if not os.path.exists(weights_path):
        print(f"Downloading pre-trained weights to {weights_path}...")
        urllib.request.urlretrieve(weights_url, weights_path)
    else:
        print("Pre-trained weights already exist.")
        
    print("--- Setup Complete ---\n")

if __name__ == "__main__":
    setup_project()
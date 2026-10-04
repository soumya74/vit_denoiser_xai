import cv2
import numpy as np
import torch

def load_image(image_path):
    """Loads an image, crops it to be compatible with SwinIR window sizes, and returns tensors."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}. Please provide a valid test image.")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Crop to a multiple of 8 (SwinIR window size requirement)
    h, w, _ = img.shape
    h, w = h - h % 8, w - w % 8
    img = img[:h, :w, :]
    
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)
    return img_tensor, img

def add_gaussian_noise(image_tensor, noise_level=50):
    """Adds synthetic Gaussian noise matching the pre-trained model's target."""
    noise = torch.randn_like(image_tensor) * (noise_level / 255.0)
    noisy_tensor = image_tensor + noise
    return torch.clamp(noisy_tensor, 0, 1)
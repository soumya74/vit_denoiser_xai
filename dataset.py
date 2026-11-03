import cv2
import torch
import numpy as np

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img.shape
    h, w = h - h % 8, w - w % 8
    img = img[:h, :w, :]
    
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor, img

def add_gaussian_noise(tensor, noise_level=50):
    noise = torch.randn_like(tensor) * (noise_level / 255.0)
    noisy_tensor = tensor + noise
    return torch.clamp(noisy_tensor, 0.0, 1.0)
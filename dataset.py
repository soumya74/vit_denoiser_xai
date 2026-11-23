import cv2
import torch
import numpy as np

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_image(image_path):
    """Existing SwinIR loader (crops to multiple of 8)."""
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


def load_image_for_mae(image_path, img_size=224):
    """MAE loader: resizes image to (img_size, img_size) in [0, 1] range."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor, img


def normalize_imagenet(tensor):
    """Applies ImageNet channel normalization expected by pre-trained MAE."""
    mean = IMAGENET_MEAN.to(tensor.device, dtype=tensor.dtype)
    std = IMAGENET_STD.to(tensor.device, dtype=tensor.dtype)
    return (tensor - mean) / std


def add_gaussian_noise(tensor, noise_level=50):
    noise = torch.randn_like(tensor) * (noise_level / 255.0)
    noisy_tensor = tensor + noise
    return torch.clamp(noisy_tensor, 0.0, 1.0)
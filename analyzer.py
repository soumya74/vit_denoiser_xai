import torch
import numpy as np

class ViTXAIAnalyzer:
    def __init__(self, model):
        self.model = model
        self.latent_features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Hooks into the Encoder and the Final Output Projection."""
        def encoder_hook(module, input, output):
            self.latent_features["encoder_out"] = output.detach().cpu()
            
        def projection_hook(module, input, output):
            self.latent_features["noise_mask"] = output.detach().cpu()
        
        # Hook 1: Deep Feature Extraction Output (End of Transformer)
        self.model.layers[-1].register_forward_hook(encoder_hook)
        
        # Hook 2: Final Output Projection (The RGB Noise Mask)
        self.model.conv_last.register_forward_hook(projection_hook)

    def analyze_drift(self, clean_tensor, noisy_tensor):
        """Runs inference and calculates spatial and channel-wise representation drift."""
        with torch.no_grad():
            _ = self.model(clean_tensor)
            z_clean = self.latent_features["encoder_out"]
            
            _ = self.model(noisy_tensor)
            z_noisy = self.latent_features["encoder_out"]
            
            # Capture the noise mask generated from the noisy image pass
            noise_mask = self.latent_features["noise_mask"]

        # 1. Spatial Drift (Cosine Distance per token)
        cos_sim = torch.nn.functional.cosine_similarity(z_clean, z_noisy, dim=-1)
        cos_dist = 1.0 - cos_sim.squeeze(0)
        
        _, _, H, W = clean_tensor.shape
        spatial_heatmap = cos_dist.view(H, W).numpy()

        # 2. Channel-Level Corruption (Mean Squared Error)
        delta_z = z_noisy - z_clean
        channel_mse = torch.mean(delta_z ** 2, dim=(0, 1)).numpy()
        top_channels = np.argsort(channel_mse)[::-1][:10]

        # This is the line that was likely missing or misaligned
        return spatial_heatmap, channel_mse, top_channels, noise_mask
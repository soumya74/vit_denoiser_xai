import torch
import numpy as np


class MAEXAIAnalyzer:
    """Extracts and compares clean vs. noisy embeddings across selected MAE encoder layers."""

    def __init__(self, model, patch_size=16, img_size=224, layer_step=3):
        self.model = model
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_layers = len(model.blocks)

        # Select every `layer_step` layer (1-indexed), always including the final layer
        self.target_layers = list(range(layer_step, self.num_layers + 1, layer_step))
        if self.num_layers not in self.target_layers:
            self.target_layers.append(self.num_layers)

        self._activations = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        for layer_num in self.target_layers:
            block_idx = layer_num - 1

            def make_hook(l_num):
                def hook(module, inp, output):
                    # Output shape: (B, 1 + N_patches, D)
                    self._activations[l_num] = output.detach().cpu()
                return hook

            handle = self.model.blocks[block_idx].register_forward_hook(make_hook(layer_num))
            self._handles.append(handle)

    def remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _encode_no_mask(self, tensor):
        self._activations.clear()
        with torch.no_grad():
            _ = self.model.forward_encoder(tensor, mask_ratio=0.0)
        return {k: v.clone() for k, v in self._activations.items()}

    def analyze_drift(self, clean_tensor, noisy_tensor):
        """
        Returns a dict keyed by layer_num (e.g., 3, 6, 9, 12) containing:
          - spatial_heatmap: (grid_size, grid_size) cosine distance per patch
          - channel_mse: (D,) mean squared error per embedding dimension
          - top_channels: top 10 most corrupted channel indices
          - mean_cos_dist: scalar mean cosine distance across all patches
          - mean_l2_dist: scalar mean L2 norm of (z_noisy - z_clean)
          - cls_cos_dist: scalar cosine distance of the [CLS] token
        """
        z_clean_map = self._encode_no_mask(clean_tensor)
        z_noisy_map = self._encode_no_mask(noisy_tensor)

        results = {}
        for layer_num in self.target_layers:
            zc = z_clean_map[layer_num]  # (1, 1 + N, D)
            zn = z_noisy_map[layer_num]

            # Separate [CLS] token and spatial patch tokens
            zc_cls, zc_patches = zc[:, 0, :], zc[:, 1:, :]
            zn_cls, zn_patches = zn[:, 0, :], zn[:, 1:, :]

            # 1. [CLS] token cosine distance
            cls_cos_dist = (1.0 - torch.nn.functional.cosine_similarity(zc_cls, zn_cls, dim=-1)).item()

            # 2. Spatial patch cosine distance
            patch_cos_sim = torch.nn.functional.cosine_similarity(zc_patches, zn_patches, dim=-1)
            patch_cos_dist = (1.0 - patch_cos_sim).squeeze(0)  # (N,)
            spatial_heatmap = patch_cos_dist.view(self.grid_size, self.grid_size).numpy()

            # 3. Channel-wise MSE & L2 drift across spatial tokens
            delta = zn_patches - zc_patches
            channel_mse = torch.mean(delta ** 2, dim=(0, 1)).numpy()  # (D,)
            top_channels = np.argsort(channel_mse)[::-1][:10]
            mean_l2_dist = torch.norm(delta, dim=-1).mean().item()

            results[layer_num] = {
                "spatial_heatmap": spatial_heatmap,
                "channel_mse": channel_mse,
                "top_channels": top_channels,
                "mean_cos_dist": float(patch_cos_dist.mean().item()),
                "mean_l2_dist": float(mean_l2_dist),
                "cls_cos_dist": float(cls_cos_dist),
            }

        return results
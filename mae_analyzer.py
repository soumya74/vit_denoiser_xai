import torch
import numpy as np


class MAEXAIAnalyzer:
    """Extracts scale-calibrated clean vs. noisy drift and clean baseline energy across MAE layers."""

    def __init__(self, model, patch_size=16, img_size=224, layer_step=3):
        self.model = model
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_layers = len(model.blocks)

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
        z_clean_map = self._encode_no_mask(clean_tensor)
        z_noisy_map = self._encode_no_mask(noisy_tensor)

        results = {}
        for layer_num in self.target_layers:
            zc_raw = z_clean_map[layer_num]  # (1, 1 + N, D)
            zn_raw = z_noisy_map[layer_num]

            # Calibrate token scale across layers via LayerNorm
            dim = zc_raw.shape[-1]
            zc = torch.nn.functional.layer_norm(zc_raw, (dim,))
            zn = torch.nn.functional.layer_norm(zn_raw, (dim,))

            zc_cls, zc_patches = zc[:, 0, :], zc[:, 1:, :]
            zn_cls, zn_patches = zn[:, 0, :], zn[:, 1:, :]

            cls_cos_dist = (1.0 - torch.nn.functional.cosine_similarity(zc_cls, zn_cls, dim=-1)).item()

            patch_cos_sim = torch.nn.functional.cosine_similarity(zc_patches, zn_patches, dim=-1)
            patch_cos_dist = (1.0 - patch_cos_sim).squeeze(0)
            spatial_heatmap = patch_cos_dist.view(self.grid_size, self.grid_size).numpy()

            delta = zn_patches - zc_patches
            mean_l2_dist = torch.norm(delta, dim=-1).mean().item()

            # Store both squared drift and clean signal variance per dimension
            raw_mse = torch.mean(delta ** 2, dim=(0, 1))
            clean_var = torch.var(zc_patches, dim=(0, 1), unbiased=False)

            # Dataset-safe floor prevents dead channels from causing false spikes
            var_floor = torch.quantile(clean_var, 0.05) + 1e-6
            rel_mse = (raw_mse / torch.maximum(clean_var, var_floor)).numpy()

            top_channels = np.argsort(rel_mse)[::-1][:10]

            results[layer_num] = {
                "spatial_heatmap": spatial_heatmap,
                "channel_mse": rel_mse,
                "top_channels": top_channels,
                "mean_cos_dist": float(patch_cos_dist.mean().item()),
                "mean_l2_dist": float(mean_l2_dist),
                "cls_cos_dist": float(cls_cos_dist),
            }

        return results
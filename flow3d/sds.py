import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from motion_prior.models.trajectory_dit_model import TrajectoriesDiT
from transformers import logging

# suppress partial model loading warning
logging.set_verbosity_error()


class SDSEdit(nn.Module):
    def __init__(self, config={}):
        super().__init__()

        self.config = config

        self.device = config.pop("device", "cuda")

        self.flow_matching_type = config.pop("flow_matching_type", "sample")
        assert self.flow_matching_type in ["sample", "v_prediction", "epsilon"], \
        f"Invalid flow matching type: {self.flow_matching_type}. Must be one of the following: sample, v_prediction, epsilon."

        self.min_step = config.pop("min_step", 20)
        self.max_step = config.pop("max_step", 980)
        self.classifier_free_scale = config.pop("classifier_free_scale", 1.0)
        self.scale_loss_by_variance = config.pop("scale_loss_by_variance", True)

        model = TrajectoriesDiT(
            max_input_length=100,
            depth=12,
            hidden_size=384,
            num_heads=6,
            context_embedding_dimension=1024,
        )
        model.to(self.device)
        self.model = model

        self.noise_scheduler = DDPMScheduler(
            prediction_type=self.flow_matching_type,
            rescale_betas_zero_snr=True,
            clip_sample=False,
        )

        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        self.variance = 1 - alphas_cumprod
        self.mean = alphas_cumprod**0.5

    def encode_tracks(self, tracks_2d):
        x = tracks_2d

        length = x.shape[2]
        if self.model.positional_encoding_num_freqs > 0:
            x = self.model.pos_embedder(x)

        embedded_x = self.model.x_embedder(x)
        embedded_x = embedded_x + self.model.pos_embed[:, :, :length]  # (B, N, T, D)

        return embedded_x

    def forward(self, tracks_2d):
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        x = self.encode_tracks(tracks_2d)

        # sample noise
        eps = torch.randn_like(x)
        x_t = self.noise_scheduler.add_noise(x, eps, t)

        # predict the noise residual with unet
        with torch.no_grad():
            eps_uncond = self.model(t, x_t)
            eps_cond = self.model(t, x_t, projected_trajectories=tracks_2d)

        eps_pred = eps_uncond + self.classifier_free_scale * (eps_cond - eps_uncond)

        w = self.mean[t]
        if self.scale_loss_by_variance:
            w = w * self.variance[t]
        loss = torch.dot(x.flatten(), w * (eps_pred - eps).flatten())
        return loss

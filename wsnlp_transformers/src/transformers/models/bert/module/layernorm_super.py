import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormSuper(torch.nn.LayerNorm):
    def __init__(self, super_embed_dim, eps=1e-12):
        super().__init__(super_embed_dim)

        self.eps = eps

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}

    def _sample_parameters(self):
        self.samples["weight"] = self.weight[: self.sample_embed_dim]
        self.samples["bias"] = self.bias[: self.sample_embed_dim]

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim

    def forward(self, x):
        self._sample_parameters()
        return F.layer_norm(
            x,
            (self.sample_embed_dim,),
            weight=self.samples["weight"],
            bias=self.samples["bias"],
            eps=self.eps,
        )

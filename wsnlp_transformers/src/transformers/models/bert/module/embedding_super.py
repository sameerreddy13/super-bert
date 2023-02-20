import torch.nn as nn
import torch.nn.functional as F


class EmbeddingSuper(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        scale=False,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

        self.super_embed_dim = embedding_dim
        self.sample_embed_dim = None
        self.sample_scale = None
        self.sample_weight = None
        self.scale = scale

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sample_weight = self.weight[..., :sample_embed_dim]
        if self.scale:
            self.sample_scale = self.super_embed_dim / sample_embed_dim

    def forward(self, x):
        x = F.embedding(
            x,
            self.sample_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        if self.scale:
            return x * self.sampled_scale
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


def sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = sample_weight[:sample_out_dim, :]
    return sample_weight


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]
    return sample_bias


class LinearSuper(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True, scale=False):

        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.sample_scale = None

        self.samples = {}

        # NOTE: Not set in calling functions.
        self.scale = scale

    def _sample_parameters(self):
        self.samples["weight"] = sample_weight(
            self.weight,
            self.sample_in_dim,
            self.sample_out_dim,
        )
        self.samples["bias"] = self.bias
        # pdb.set_trace() # TODO remove
        # print("INSIDE LINEAR SUPER", self.super_out_dim, self.sample_out_dim)
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples["bias"] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        """
        Set config for sample submodule
        """
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

    def forward(self, x):
        self._sample_parameters()
        output = F.linear(x, self.samples["weight"], self.samples["bias"])
        return output * (self.sample_scale if self.scale else 1)

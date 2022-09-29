# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT Super model configuration """
from collections import OrderedDict
from typing import Mapping

from .configuration_bert import BertConfig
from ...utils import logging

logger = logging.get_logger(__name__)

class SuperBertConfig(BertConfig):
    r"""
    TODO @sameer - update docstring

    This is the configuration class to store the configuration of a :class:`~transformers.BertModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a BERT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        classifier_dropout (:obj:`float`, `optional`):
            The dropout ratio for the classification head.

    Examples::

        >>> from transformers import BertModel, BertConfig

        >>> # Initializing a BERT bert-base-uncased style configuration
        >>> configuration = BertConfig()

        >>> # Initializing a model from the bert-base-uncased style configuration
        >>> model = BertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        intermediate_size=3072,             
        hidden_act="gelu",
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        # @sameer -  args for super net
        mlp_ratio=4,                        # ratio of mlp hidden size to embedding size
        embed_dim=768,                      # embedding size. default to hidden_size
        hidden_dropout_prob=0.1,            # becomes super_dropout
        attention_probs_dropout_prob=0.1,   # becomes super_attn_dropout
        num_hidden_layers=12,               # becomes super_num_layers
        num_attention_heads=12,             # becomes super_num_heads
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

        # the configs of super arch
        self.super_layer_num = num_hidden_layers
        self.super_embed_dim = embed_dim
        self.super_mlp_ratio = mlp_ratio
        self.super_num_heads = num_attention_heads
        self.super_dropout = hidden_dropout_prob
        self.super_attn_dropout = attention_probs_dropout_prob
        self.super_ffn_embed_dim = int(mlp_ratio * embed_dim)
        # self.num_classes = num_classes
        # self.pre_norm=pre_norm
        # self.scale=scale
        # self.gp = gp

        # configs for the sampled subTransformer
        self.sample_out_dim = None
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_attn_dropout = None
        self.sample_output_dim = None
        self.sample_ffn_embed_dim = None


class BertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("last_hidden_state", {0: "batch", 1: "sequence"}), ("pooler_output", {0: "batch"})])


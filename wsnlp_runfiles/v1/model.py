import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizerFast, BertForMaskedLM, BertConfig
import torch

import math
#### Main Functions ####
def get_subnet(model, depth, attention_ratio, intermediate_ratio, hidden_dimension,preserve_finetune_hidden_size=False):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        
    if 0 < attention_ratio <= 1.0:
        model.bert.set_attention_ratio(attention_ratio)

    if 0 < intermediate_ratio <= 1.0:
        model.bert.set_intermediate_ratio(intermediate_ratio)

    if depth > 0:
        model.bert.set_max_layers(depth)
    
    if hidden_dimension > 0:
        preserve_finetune_hidden_size=True
        model.set_hidden_dimension(hidden_dimension,preserve_finetune_hidden_size=preserve_finetune_hidden_size)


    return model

       

def load_tokenizer(model_key):
    return BertTokenizerFast.from_pretrained(model_key)

def load_model(
        model_key, 
        task=None, 
        pretrain_chkpt=None,
        task_config=None,
        select_layers='',
    ):
    # Get model
    if task == 'pretrain':
        model = BertForMaskedLM.from_pretrained(model_key)
    else:
        model = BertForSequenceClassification.from_pretrained(model_key, config=task_config)

    # Load pretrain weights
    if pretrain_chkpt:
        model.load_state_dict(torch.load(pretrain_chkpt), strict=False)

    # Set select layers method
    model.bert.set_select_layers_method(select_layers)

    return model



def disable_dropout(model):
    def disable_drop(m):
        if type(m) == nn.Dropout:
            m.eval()

    model.apply(disable_drop)

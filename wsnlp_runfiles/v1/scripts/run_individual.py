from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()
import torch
import torch.utils.data as torch_data
import constants
import utils
import datasets
import numpy as np
from tqdm import tqdm
import pdb
from pathlib import Path
import os
from train import train_and_eval_model
import argparse
import pprint
from data import fetch_sst2_train_val_test
import wandb


BASE_DIR = Path(__file__).parent / 'outputs_run_individual'
def get_outputs_dir(id=constants.DEFAULT_ID) -> Path:
    return BASE_DIR / id
    
def get_hidden_dim_output_dir(id=constants.DEFAULT_ID) -> Path:
    return get_outputs_dir(id) / 'vary_hidden_dim'

def get_layer_output_dir(id=constants.DEFAULT_ID) -> Path:
    return get_outputs_dir(id) / 'vary_layer'

def get_miniature_output_dir(id=constants.DEFAULT_ID) -> Path:
    return get_outputs_dir(id) / 'vary_miniature'

def get_attention_output_dir(id=constants.DEFAULT_ID, attention_approach=None) -> Path:
    assert attention_approach is not None, "Specify the approach for elastic attention"
    return get_outputs_dir(id) / ('vary_attention_'+attention_approach)

def get_attention_model_names(attention_approach):
    pretrained=args.base_model
    attention_ratio_list = args.attention_config
    prefix = "att_"+attention_approach+"_"
    custom_log_file_names = [pretrained+"_attention_" + str(attention_approach) + "_" + str(i) for i in attention_ratio_list]
    model_names=[pretrained]*len(custom_log_file_names)
    print(custom_log_file_names)
    return model_names,custom_log_file_names, attention_ratio_list

def get_elastic_layer_output_dir(id=constants.DEFAULT_ID, ) -> Path:
    return get_outputs_dir(id) / 'vary_layer_elastic'

def get_elastic_layer_model_names():
    pretrained=args.base_model
    # layer_list=[i for i in range(2,13,2)]
    layer_list = args.layer_config
    custom_log_file_names = [pretrained +"_layer" + "_" + str(i) for i in layer_list]
    model_names=[pretrained]*len(custom_log_file_names)
    print(custom_log_file_names)
    return model_names,custom_log_file_names, layer_list

def get_elastic_hidden_model_names():
    pretrained=args.base_model
    # layer_list=[i for i in range(2,13,2)]
    hidden_list = args.hidden_config
    custom_log_file_names = [pretrained +"_layer" + "_" + str(i) for i in hidden_list]
    model_names=[pretrained]*len(custom_log_file_names)
    print(custom_log_file_names)
    return model_names,custom_log_file_names, hidden_list

def get_joint_model_names():
    pretrained=args.base_model
    assert len(args.layer_config) == len(args.attention_config), "the number of layer and attention configuration should match"
    assert len(args.hidden_config) == len(args.attention_config), "the number of hidden and attention configuration should match"
    custom_log_file_names = []
    for i in range(len(args.layer_config)):
        custom_log_file_names.append(pretrained+"_layer_" + str(args.layer_config[i])+"_attention_"+str(args.attention_config[i])+"_hidden_"+str(args.hidden_config[i]))
    model_names=[pretrained]*len(custom_log_file_names)
    return model_names, custom_log_file_names

def get_joint_output_dir(id=constants.DEFAULT_ID) -> Path:
    return get_outputs_dir(id) / 'joint'



# Training
def train_models(model_names, output_dir, train_kwargs, custom_log_file_names=[],attention_config=[],layer_config=[],hidden_config=[],repeat=3):
    if custom_log_file_names:
        assert len(model_names) == len(custom_log_file_names)
    eval_stats = {}   
    train_kwargs['epochs']=args.epochs
    train_kwargs['eval_stats'] = eval_stats    
    train_kwargs['output_dir'] = output_dir
    print("Writing experiment results to:", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i, m in enumerate(model_names):
        eval_stats[m] = []
        print("Training:", m)
        for r in range(repeat):
            print("Trial ",r)
            train_kwargs['model_key'] = m
            if custom_log_file_names:
                train_kwargs['custom_log_file_name'] = custom_log_file_names[i]
            if len(attention_config) > 0:
                train_kwargs['elastic_attention']=True
                train_kwargs['attention_ratio']=attention_config[i]
            if len(layer_config):
                train_kwargs['elastic_layer']=True
                train_kwargs['depth']=layer_config[i]
            if len(hidden_config):
                train_kwargs['elastic_hidden']=True
                train_kwargs['hidden_dimension']=hidden_config[i]
            train_and_eval_model(**train_kwargs)
        pprint.pp(eval_stats[m])
    pprint.pp(eval_stats)
    for model, accs in eval_stats.items():
        print(model, np.mean(accs[0]))

def parse_args():
    parser = argparse.ArgumentParser(description='Run individual model training on SST2 for different elasticities', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', type=int, help="GPU id")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--layer', action="store_true", help="Run for varying layer num")
    # parser.add_argument('--elastic_layer', action="store_true", help="Run for varying layer num")

    parser.add_argument('--hidden', action="store_true", help="Run for varying hidden dim")
    parser.add_argument('--miniature', action="store_true", help="Run for varying bert miniatures")
    # parser.add_argument('--attention', action="store_true", help="Run for varying bert attention ratio")
    parser.add_argument('--attention_approach', choices=[ "head_num", "head_size_1", "head_size_2"], default=None, action="append", help="approach for elastic attention layer [ head_num, head_size_1, head_size_2]")
    parser.add_argument('-a', "--all", action="store_true", help="Run for all variations")
    parser.add_argument('--limit', type=int, default=-1, help="Limit the number of training samples used")
    parser.add_argument('-w', '--wandb', action="store_true", help="Use wandb logging")
    parser.add_argument('--wandb_user', type=str, default=None, 
        help="wandb user entity")

    parser.add_argument('--wandb_project', type=str, default=None, 
        help="wandb user entity")
    parser.add_argument('--save_dir', type=str, default=constants.DEFAULT_ID, help="Specify id for subfolder to save results to. Writes to 'outputs.../id'")
    parser.add_argument('--save_model', action="store_false", default=True, help="Save trained models")
    parser.add_argument('--val_split', type=float, default=0.05,
        help="Split size for train val split of sst2 data")
    parser.add_argument('--attention_config', nargs='+', type=float,default=None,
                    help='list of attention ratio values provided as a space separated list\n ex) 0.1 0.2 0.3\n each value must be a value between (0,1]')
    parser.add_argument('--layer_config', nargs='+', type=int,default=None,
                    help='list of layer number provided as a space separated list\n ex) 2 4 6 8\ each value must be a value between [1, maximum model layer]')
    parser.add_argument('--hidden_config', nargs='+', type=int, default=None,
                    help='''Elastic hidden configuration''')
    parser.add_argument('--base_model',type=str, default='bert-large-uncased', help="the pretrained supernet that would be trained")

    parser.add_argument('--epochs', type=int, default=5, help="number of epochs")

    parser.add_argument('--lr', type=float, default=1e-5, 
        help="Learning rate") 
    scheduler_group = parser.add_mutually_exclusive_group()
    scheduler_group.add_argument('--constant_schedule', action='store_true',
        help='Constant learning rate schedule')

    scheduler_group.add_argument('--linear_schedule_with_warmup', action='store_true',
        help='Linear increase and decay learning rate scheduler. See `get_linear_schedule_with_warmup` from huggingface')

    scheduler_group.add_argument('--cosine_schedule_with_warmup', action='store_true',
        help='Linear increase and cosine decay. See transformers.get_cosine_schedule_with_warmup')

    parser.add_argument('--warmup', type=float, default=0.1,
        help="Warmup percentage if using a scheduler with warmup steps. E.g. set --warmup=0.1 to warmup for first 10 percent of training")
    
    parser.add_argument('--clip_gradient', action='store_true', default=False,
        help='Enable gradient clipping')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # Config
    device = utils.get_device(args.gpu)
    utils.seed_everything(constants.SEED)
    output_dir = get_outputs_dir(args.save_dir)
    # Start script
    print("DEVICE:", device)
    utils.overwrite_dir(output_dir)
    print("Created destination folder:", output_dir)
    # Get data
    # raw_datasets = datasets.load_dataset("glue", "sst2")
    # train_dataset, test_dataset = raw_datasets['train'], raw_datasets['validation']
    train, val, test, split_idx_dict = fetch_sst2_train_val_test(args.limit, args.val_split)
    if args.limit > 0:
        train_dataset = train_dataset.select(range(args.limit))
        test_dataset = test_dataset.select(range(args.limit))

    if args.wandb:
        assert args.wandb_user is not None, "Specify wandb user name to use wandb"
        assert args.wandb_project is not None,"Specify wandb project name to use wandb"
        wandb.init(project= args.wandb_project,
            entity=args.wandb_user,
            name=args.save_dir
            )


    train_kwargs = {
        'model_key':                None,
        'train_dataset':            train,
        'val_dataset':              val,
        'test_dataset':             test,
        'output_dir':               None,
        'device':                   device,
        'eval_stats':               None,
        'use_wandb':                args.wandb,
        'batch_size':               args.batch_size,
        'custom_log_file_name':     None,
        'save_model':               args.save_model,
        'num_epochs':               args.epochs,
        'learning_rate':            args.lr,
        'constant_schedule':            args.constant_schedule,
        'linear_schedule_with_warmup':  args.linear_schedule_with_warmup,
        'cosine_schedule_with_warmup':  args.cosine_schedule_with_warmup,
        'warmup':                       args.warmup,
        'clip_gradient':                args.clip_gradient,
    }
    # Training
    if args.layer or args.all:
        print("VARYING LAYER NUM")
        l, o = constants.VARY_ENCODER_LAYER, get_layer_output_dir(id=args.save_dir)
        train_models(l, o, train_kwargs)
    if args.hidden or args.all: 
        print("VARYING HIDDEN DIM")
        l, o = constants.VARY_HIDDEN_DIM, get_hidden_dim_output_dir(id=args.save_dir)
        train_models(l, o, train_kwargs)
    if args.miniature or args.all:
        print("VARYING MINIATURES")
        l, o, c = constants.VARY_MINIATURES.values(), get_miniature_output_dir(id=args.save_dir), list(constants.VARY_MINIATURES.keys())
        train_models(l, o, train_kwargs, custom_log_file_names=c)
    if args.attention_config is not None and args.layer_config is not None and args.hidden_config is not None:
        l,custom_log_file_names = get_joint_model_names()
        o = get_joint_output_dir(id=args.save_dir)
        train_models(l,o,train_kwargs,custom_log_file_names=custom_log_file_names,layer_config=args.layer_config, attention_config=args.attention_config, hidden_config=args.hidden_config)
    elif args.attention_config is not None:
        l,custom_log_file_names,attention_config = get_attention_model_names(args.attention_approach[0])
        o = get_attention_output_dir(id=args.save_dir,attention_approach=args.attention_approach[0])
        train_models(l, o, train_kwargs,custom_log_file_names=custom_log_file_names, attention_config=attention_config)
    elif args.layer_config is not None:
        l,custom_log_file_names, layer_config= get_elastic_layer_model_names()
        o = get_elastic_layer_output_dir(id=args.save_dir)
        train_models(l, o, train_kwargs,custom_log_file_names=custom_log_file_names,layer_config=layer_config)
    elif args.hidden_config is not None:
        l,custom_log_file_names, hidden_config= get_elastic_hidden_model_names()
        o = get_elastic_layer_output_dir(id=args.save_dir)
        train_models(l, o, train_kwargs,custom_log_file_names=custom_log_file_names,hidden_config=hidden_config)

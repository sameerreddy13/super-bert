import numpy as np
import torch
from transformers import logging
logging.set_verbosity_error()

import os
from os import makedirs
from os.path import join, abspath, dirname, isdir
import argparse
from constants import SEED, RAMP_UP, SYNC, VERBOSE, DEFAULT_ID
from constants import VARY_ENCODER_LAYER, VARY_HIDDEN_DIM, VARY_MINIATURES
import utils
from latency import *
from data import load_data
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent / 'outputs_run_perf'
def get_outputs_dir(id) -> Path:
    return BASE_DIR / id

def parse_args():
    parser = argparse.ArgumentParser(description="Latency measurements for individual pre-trained models", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to run each model for")
    parser.add_argument("--trials", type=int, default=3, help="Number of repeated trials for each model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument('--layer', action="store_true", help="Run for varying layer num")
    parser.add_argument('--hidden', action="store_true", help="Run for varying hidden dim")
    parser.add_argument('--miniature', action="store_true", help="Run for varying bert miniatures")
    # parser.add_argument('--attention', action="store_true", help="Run for varying bert attention ratio")
    parser.add_argument('--attention_approach', choices=[ "head_num", "head_size_1", "head_size_2"], default=None, action="append", help="approach for elastic attention layer [ head_num, head_size_1, head_size_2]")
    parser.add_argument('-a', "--all", action="store_true", help="Run for all variations")
    parser.add_argument('--id', type=str, default=DEFAULT_ID, help="Specify id for subfolder to save results to. Write to 'outputs.../id'")
    parser.add_argument('--attention_config', nargs='+', type=float,default=None,
                    help='list of attention ratio values provided as a space separated list\n ex) 0.1 0.2 0.3\n each value must be a value between (0,1]')
    parser.add_argument('--layer_config', nargs='+', type=int,default=None,
                    help='list of layer number provided as a space separated list\n ex) 2 4 6 8\ each value must be a value between [1, maximum model layer]')
    parser.add_argument('--hidden_config', nargs='+', type=int, default=None,
                    help='list of hidden size number provided as a space separated list\n ex) 128 256 512 768 1024\ ')
    parser.add_argument('--base_model',type=str, default='bert-large-uncased', help="the pretrained supernet that would be trained")

    return parser.parse_known_args()

# def dirname():
#     return abspath(dirname(__file__))

if __name__ == '__main__':
    args, _ = parse_args()
    # Config
    # source_folder = join(dirname(), '..', 'data')
    print(args)
    source_folder = str(Path(__file__).parent.parent.parent / 'data')
    device = utils.get_device(args.gpu)
    utils.seed_everything(SEED)
    output_dir = get_outputs_dir(args.id)
    if SYNC:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # Start script
    print(device)

    assert isdir(source_folder), "please set up the correct source folder directory" + str(source_folder)
    utils.overwrite_dir(output_dir)
    print("Created destination folder : ", output_dir)
    print("Starting the run script")
    # Setup inputs
    latency_kwargs = dict(steps=args.steps, trials=args.trials, batch_size=args.batch_size)
    print("Parameters for latency experiment", latency_kwargs)
    trials = latency_kwargs.pop('trials')
    test_item = load_data(source_folder, device, batch_size=latency_kwargs.pop('batch_size'))
    if VERBOSE:
        print(test_item.shape)
    # Run

    for i in range(1, trials+1):
        log_dir = utils.create_log_dir(i, output_dir)
        print("measuring for", log_dir)
        # measure_simulated_miniature(test_item, source_folder, device, log_dir,"bert-large-uncased", **latency_kwargs)
 
        if args.miniature or args.all:
            measure_latency_for_miniatures(test_item, source_folder, device, log_dir, **latency_kwargs)
        if args.layer or args.all:
            measure_latency_along_list(test_item, source_folder, device, log_dir, VARY_ENCODER_LAYER, **latency_kwargs)
        if args.hidden or args.all:
            measure_latency_along_list(test_item, source_folder, device, log_dir, VARY_HIDDEN_DIM, **latency_kwargs)
        if args.attention_config is not None and args.layer_config is not None and args.hidden_config is not None:
            measure_end_to_end_joint(test_item, source_folder, device, log_dir, args.layer_config,args.attention_config,args.hidden_config, base_model=args.base_model, **latency_kwargs)
        elif args.attention_config:
            assert args.attention_approach is not None, "please specify the elastic attention approach using the keyword --approach"
            measure_end_to_end_varying_attention_ratio(test_item, source_folder, device, log_dir, attention_config=args.attention_config,base_model=args.base_model, **latency_kwargs)
        elif args.layer_config:
            measure_end_to_end_varying_elastic_layer(test_item,source_folder,device,log_dir,layer_config=args.layer_config,base_model=args.base_model, **latency_kwargs)
        elif args.hidden_config:
            measure_end_to_end_varying_elastic_hidden(test_item,source_folder,device,log_dir,hidden_config=args.hidden_config,base_model=args.base_model, **latency_kwargs)
        

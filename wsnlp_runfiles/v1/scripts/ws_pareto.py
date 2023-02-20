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
import ws_train
import train
import argparse
import pprint
import pickle
import os
import wandb
from latency import *
from model import load_model_and_tokenizer, get_subnet
# from ws_train import generate_elastic_configs,ElasticConfig
from ws_train import ElasticConfig
import pandas as pd
import data
from train_utils import eval_model



BASE_DIR = Path(__file__).parent / 'outputs_run_ws'
def get_outputs_dir(id:constants.DEFAULT_ID) -> Path:
    return BASE_DIR / id

def parse_args():
    parser = argparse.ArgumentParser(description="wsPareto - get acc and latency from model. Use saved model from experiment directory ", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_dir', type=str, default=constants.DEFAULT_ID, 
        help="Specify path for subfolder to save results to. Writes to 'outputs.../exp_dir'")

    parser.add_argument('--gpu', type=int, 
        help="GPU id")

    parser.add_argument('--checkpt_epoch', type=int, 
        help="best checkpoint epoch")

    parser.add_argument('--trial_number', type=int, 
        help="trial number to load the model")

    parser.add_argument('-w', '--wandb', action="store_true", 
        help="Use wandb logging")

    parser.add_argument('--wandb_user', type=str, default=None, 
        help="wandb user entity")

    parser.add_argument('--wandb_project', type=str, default=None, 
        help="wandb user entity")

    parser.add_argument("--load_acc",action="store_true",
        help="load accuracy from a checkpoint")

    parser.add_argument('--basenet',type=str, default='bert-large-uncased', 
        help="The pretrained model key used for training")

    parser.add_argument('--elastic_width', nargs='+', type=float, default=None, 
        help='''[NOT IMPLEMENTED] Elastic width configuration, provided as a space separated list i.e. 0.1 0.2 0.3.\n
                Jointly adjusts attention and intermediate with a single width multiplier.\n
                Each value must be a value between (0,1].''')

    parser.add_argument('--elastic_attention', nargs='+', type=float, default=constants.ELASTIC_ATTENTION_CONFIG, 
        help='Elastic attention configuration')

    parser.add_argument('--elastic_intermediate', nargs='+', type=float, default=constants.ELASTIC_INTERMEDIATE_CONFIG, 
        help='Elastic intermediate configuration')

    parser.add_argument('--elastic_layer', nargs='+', type=int, default=constants.ELASTIC_LAYER_CONFIG,
        help='''Elastic layer configuration i.e. 2 4 6 8.\n
                Each value must be a value between [1, maximum model layer]''')
    
    parser.add_argument('--elastic_hidden', nargs='+', type=int, default=constants.ELASTIC_HIDDEN_CONFIG,
        help='''Elastic hidden configuration''')
    
    parser.add_argument('--batch_size', type=int, default=16, 
        help="Batch size")

    parser.add_argument("--steps", type=int, default=500, help="Number of steps to run each model for")

    parser.add_argument('--coupling', action="store_true",
        help="Leverage coupling between elastic layer and elastic attention ratio")
    return parser.parse_args()

    




if __name__ == '__main__':
    args = parse_args()
    # Config
    device = utils.get_device(args.gpu)
    # utils.seed_everything(constants.SEED)
    output_dir = get_outputs_dir(id=args.exp_dir)

    # Start script
    print("\nDEVICE:", device)

    # load the training configuration
    with open(output_dir / 'config.pkl','rb') as f:
        train_config = pickle.load(f)


    model = BertForSequenceClassification.from_pretrained(train_config["basenet"], num_labels=2)

    # load the acc data
    if args.load_acc:
        basename="train"
        epoch_metrics = torch.load(output_dir / f'train_epoch_{args.checkpt_epoch}_metrics.pt')

    else:
        # load model
        print("Loading model")
        model.to(device)
        model.load_state_dict(torch.load(output_dir / "model_weights"/ f'final_epoch10_weights.pt'))
        model.eval()

        # print elastic configurations
        elastic_configs, eval_configs = utils.generate_elastic_configs(args)
        # prepare evaluation dataset
        test_dataset=datasets.load_dataset("glue", "sst2")['validation']
        tokenizer = BertTokenizerFast.from_pretrained(train_config["basenet"])
        test_loader = data.sst_loader(tokenizer, test_dataset, args.batch_size)
    

    # # load latency dataset
    # source_folder = str(Path(__file__).parent.parent.parent / 'data')
    # test_item = load_data(source_folder, device, batch_size=128)
    
    # log_table={}

    # prepare iteration.
    # if load -> iter through final_epoch_metrics
    # if not load -> iter through the elastic configs

    iter_list = epoch_metrics["subnet_test_accs"] if args.load_acc else eval_configs

    print("iterating through")
    print(iter_list)
    #TODO : iterate through all the generated models.
    for item in iter_list:
        if args.load_acc:
            config,test_acc = item
            config = ElasticConfig.from_name(config)
            acc=test_acc
        else:
            config=item

        print("running : ",config)
        child = get_subnet(
                model, 
                config.depth, 
                config.attention_ratio, 
                config.intermediate_ratio)
        
        #meausre accuracy
        acc = test_acc if args.load_acc else eval_model(child, test_loader)
        print("acc : ",acc)
        #measure latency.
        measure_latency(child, test_item, device, open("remove.log","w"), args.steps)
        data_file = pd.read_csv("remove.log",header=None)[0].tolist()
        data = [x for x in data_file]
        latency = np.percentile(data, constants.PERCENTILE)
        print(acc, latency)

        # log
        config=config.get_name()
        with open(args.save_dir+"_log_pareto_data.log","a") as f:
            f.write(f"{config} {acc} {latency}\n")
        log_table[config]=(acc,latency)

    # sort based on the latency value
    pareto_data = list(log_table.items())
    pareto_data.sort(key=lambda x: x[1][1])
    torch.save(pareto_data,f"pareto_data_{args.save_dir}.pt")
    pprint.pp(pareto_data)
    # measure latency
    pickle.dump(
        pareto_data,
        (f"pareto_data_{args.save_dir}.pt").open('wb'))


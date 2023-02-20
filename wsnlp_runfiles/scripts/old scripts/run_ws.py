from transformers import logging
logging.set_verbosity_error()
import torch
import torch.utils.data as torch_data
import constants
import utils!@
import datasets
import numpy as np
from tqdm import tqdm
import pdb
from pathlib import Path
import os
import ws_train
import train as train_single
import argparse
import pprint
import pickle
import os
import data as src_data
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run elastic training \n Select either one of '--individual' or '--ws_sandwich' as training options", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ##################################### Required #####################################
    parser.add_argument('--task', 
        type=str, 
        required=True,
        help="Task the model is trained on")

    parser.add_argument('--output_dir', 
        type=str, 
        default=constants.DEFAULT_ID, 
        help="Specify path for subfolder to save results to. Writes to 'outputs.../output_dir'")

    parser.add_argument('--train_type',
        type=str,
        required=True,
        choices=['individual', 'ws_sandwich'])

    ################################### Device ###################################
    device_group = parser.add_mutually_exclusive_group()

    device_group.add_argument('--gpu', 
        type=int, 
        help="GPU id")

    device_group.add_argument('--cpu', 
        action='store_true',
        help="Use cpu")

    device_group.add_argument('--distributed',
        action="store_true",
        help="Data parallel training")

    ##################################### Data attributes #####################################
    parser.add_argument('--limit', 
        type=int, 
        default=-1, 
        help="Limit the number of training samples used")

    parser.add_argument('--val_split', 
        type=float, 
        default=0.05,
        help="Split size for train val split of sst2 data")

    ##################################### Model saving #####################################
    parser.add_argument('--no_model_save', 
        action="store_true",
        help="Don't save any model weights")

    parser.add_argument('--save_best_only', 
        action='store_true',
        help='save only the best model')

    ## ################################### model init #####################################
    parser.add_argument('--pretrain_chkpt', 
        type=str,
        help="The pretrained model used for initialization."\
        "This can be a model key in the huggingface repository or a path to saved model weights.")

    parser.add_argument('--pretrain_tokenized_path', 
        type=str, 
        default='bookcorpus_tokenized/',
        help="Path to tokenized pretrain dataset")

    parser.add_argument('--basenet',
        type=str, 
        default='bert-large-uncased', 
        help="The pretrained model key used for training")


    ##################################### Training loop settings #####################################
    parser.add_argument('--max_seq_length', 
        type=int, 
        default=128,
        help='Max sequence length for model')
    
    parser.add_argument('--batch_size', 
        type=int, 
        default=64, 
        help="Batch size")

    parser.add_argument('--epochs', 
        type=int, 
        default=10,
        help="Number of epochs")

    parser.add_argument('--max_train_steps',
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides epochs argument. One training step == one update step i.e. optimizer.step()")

    parser.add_argument('--eval_every_num_steps',
        type=int, 
        default=None,
        help='Eval every N steps on validation and test sets. If not provided then eval occurs after each epoch instead.')

    ##################################### Weight shared training parameters #####################################
    parser.add_argument('--select_layers',
        type=str,
        default='bottom',
        choices=['uniform', 'top', 'bottom', 'everyother'],
        help='Method for selecting layers when choosing a subnet of custom depth')

    parser.add_argument('--num_subnets', 
        type=int,
        default=2, 
        help="Number of subnets to sample for sandwich sampling")

    parser.add_argument('--alpha', 
        type=float, 
        default=1.0, 
        help="The weight of distillation loss." +
        "If alpha is 0.0 then there is no distillation. If alpha is 1.0 then there is full distillation.") 

    parser.add_argument('--regularize_once', 
        action="store_true",
        help="Regularize the largest model only like in BigNAS")

    parser.add_argument('--coupling', 
        action="store_true",
        help="Leverage coupling between elastic depth and elastic attention ratio")

    ####### Elastic dimensions. 
    ####### Note, you can set a given dimension argument to -1 to leave it unmodified in weight shared training. 
    ####### e.g. `python run_ws.py ... --elastic_hidden -1`
    parser.add_argument('--elastic_width', 
        nargs='+', 
        type=float, 
        default=None, 
        help='''Elastic width configuration, provided as a space separated list i.e. 0.1 0.2 0.3.\n
                Jointly adjusts attention and intermediate with a single width multiplier.\n
                Each value must be a value between (0,1].''')

    parser.add_argument('--elastic_attention', 
        nargs='+', 
        type=float, 
        default=constants.ELASTIC_ATTENTION_CONFIG, 
        help='Elastic attention configuration')

    parser.add_argument('--elastic_intermediate', 
        nargs='+', 
        type=float, 
        default=constants.ELASTIC_INTERMEDIATE_CONFIG, 
        help='Elastic intermediate configuration')

    parser.add_argument('--elastic_depth', 
        nargs='+', 
        type=int, 
        default=constants.ELASTIC_LAYER_CONFIG,
## Training loop settings ##
        help='''Elastic depth configuration i.e. 2 4 6 8.\n
                Each value must be a value between [1, maximum model depth]''')

    parser.add_argument('--preserve_finetune_hidden_size', 
        action="store_true",
        help="Preserve the finetune head hidden size")

    parser.add_argument('--elastic_hidden', nargs='+', type=int, default=constants.ELASTIC_HIDDEN_CONFIG,
        help='''Elastic hidden configuration''')


    ##################################### Learning rate and hyperparams #####################################
    parser.add_argument('--clip_gradient', 
        action='store_true', 
        default=False,
        help='Enable gradient clipping')

    parser.add_argument('--lr', 
        type=float, 
        default=5e-6, 
        help="Learning rate") 

    scheduler_group = parser.add_mutually_exclusive_group()

    scheduler_group.add_argument('--constant_schedule', 
        action='store_true',
        help='Constant learning rate schedule')

    scheduler_group.add_argument('--linear_schedule_with_warmup',
        action='store_true',
        help='Linear increase and decay learning rate scheduler. See `get_linear_schedule_with_warmup` from huggingface')

    scheduler_group.add_argument('--cosine_schedule_with_warmup', 
        action='store_true',
        help='Linear increase and cosine decay. See transformers.get_cosine_schedule_with_warmup')

    parser.add_argument('--warmup', 
        type=float, 
        default=0.1,
        help="Warmup percentage if using a scheduler with warmup steps. E.g. set --warmup=0.1 to warmup for first 10 percent of training")

    ##################################### Other #####################################

    parser.add_argument('--seed',
        type=int,
        default=None,
        help="Seed")

    parser.add_argument('--trials', 
        type=int, 
        default=1, 
        help="Average over x trials. NOTE: Only used for weight shared training")

    ##################################### Logging #####################################
    parser.add_argument('-w', '--wandb', 
        action="store_true", 
        help="Use wandb logging")

    parser.add_argument('--wandb_user', 
        type=str, 
        default=None, 
        help="wandb user entity")

    parser.add_argument('--wandb_project', 
        type=str, 
        default=None, 
        help="wandb user entity")

    parser.add_argument('--disable_tqdm',
        action="store_true", 
        help='Turn on to disable tqdm progress bar for train loop')

    return parser.parse_args()

######### Functions #########    

BASE_DIR = Path(__file__).parent / 'outputs_run_ws'

def get_outputs_dir(id=constants.DEFAULT_ID) -> Path:
    return BASE_DIR / id

def save_args(args, output_dir):
    save_config = vars(args)
    pickle.dump(
        save_config,
        (output_dir / 'config.pkl').open('wb'))

    cmd_elems = sys.argv
    cmd_elems[0] = 'python -m scripts.run_ws'
    cmd_line = ' '.join(cmd_elems).strip()
    with (output_dir / 'command.txt').open('w') as fp:
        fp.write(cmd_line)


def save_train_val_idx(output_dir, idx_dict):
    pickle.dump(
        idx_dict,
        (output_dir / 'train_val_idx.pkl').open('wb'))

def train_individual(output_dir, config, train_kwargs):
    c = config
    print(f"Training model ({c.get_name()})")            
    indiv_dir = output_dir / c.get_name() 
    os.makedirs(indiv_dir)
    train_kwargs.update({
        'depth':                c.depth,
        'attention_ratio':      c.attention_ratio,              
        'intermediate_ratio':   c.intermediate_ratio,
        'hidden_dimension':     c.hidden_dimension,
        'output_dir':           indiv_dir,
    })
    train_single.train_and_eval_model(**train_kwargs)

def set_train_data(args, train_kwargs, output_dir):
    '''
    Helper to get a train val split, save the index dict and update training arguments
    '''
    # Get data and train val test splits
    print("-"*20)
    config = None
    supported_tasks = {'sst2', 'mnli', 'pretrain'}
    assert args.task in supported_tasks, "the given task is not defined"
    
    # Pretrain
    if args.task == "pretrain":
        train, val, test, _ = src_data.fetch_pretrain_train_val_test(args.pretrain_tokenized_path, args.limit, args.val_split)
    # Assume glue task if not pretrain
    else:
        train, val, test, idx_dict, config = src_data.fetch_glue_train_val_test(args.basenet, args.task, args.limit, args.val_split)
        # Save train val split indices
        save_train_val_idx(output_dir, idx_dict)

    print("Loaded datasets")
    print("Train:", train)
    print(f"Val (from {args.val_split} split of train set):", val)
    print("Test (from task GLUE dev set):", test)
    print("-"*20)

    # Return updated train kwargs
    update_dict = {
        'train_dataset':    train,
        'val_dataset':      val,
        'test_dataset':     test,
        'task_config':      config,
    }
    return {**train_kwargs, **update_dict}


if __name__ == '__main__':

    ################### Setup ###################

    args = parse_args()


    # Configure device
    device = utils.get_device(args.gpu)
    if args.distributed:
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.cpu:
        device = 'cpu'

    # Get directory for experiment outputs
    output_dir = get_outputs_dir(id=args.output_dir)

    # Start script
    print("\nDEVICE:", device, "SEED:", args.seed)

    # Create output directory(s)
    utils.overwrite_dir(output_dir)

    # Seed
    if args.seed is not None:
        utils.seed_everything(args.seed)

    # Setup wandb
    if args.wandb:
        assert args.wandb_user is not None, "Specify wandb user name to use wandb"
        assert args.wandb_project is not None,"Specify wandb project name to use wandb"
        wandb.init(project= args.wandb_project,
            entity=args.wandb_user,
            name=args.output_dir)
    if args.pretrain_chkpt:
        # TODO - allow for using custom weights file, not just this specific validation checkpoint
        pretrain_chkpt = get_outputs_dir(args.pretrain_chkpt) /"0"/"model_weights"/"epoch_0_step_1000_best_val_weights.pt"
    else:
        pretrain_chkpt = None

    # Create train config
    train_kwargs = {
        'elastic_configs':              None,
        'device':                       device,
        "pretrain_chkpt":               pretrain_chkpt,         # TODO - add support for individual training
        'output_dir':                   output_dir,
        'save_model':                   not args.no_model_save,
        'model_key':                    args.basenet,
        'num_epochs':                   args.epochs,
        'max_train_steps':              args.max_train_steps,
        'eval_every_num_steps':         args.eval_every_num_steps,
        'batch_size':                   args.batch_size,
        'learning_rate':                args.lr,
        'use_wandb':                    args.wandb,
        'alpha':                        args.alpha,
        'num_subnets':                  args.num_subnets,
        'constant_schedule':            args.constant_schedule,
        'linear_schedule_with_warmup':  args.linear_schedule_with_warmup,
        'cosine_schedule_with_warmup':  args.cosine_schedule_with_warmup,
        'warmup':                       args.warmup,
        'regularize_once':              args.regularize_once,
        'clip_gradient':                args.clip_gradient,
        "save_best_only":               args.save_best_only,
        "pretrain_chkpt":               pretrain_chkpt, # TODO - add support for individual training
        "task":                         args.task,           # TODO - add support for individual training
        "preserve_finetune_hidden_size": args.preserve_finetune_hidden_size,
        "max_seq_length":               args.max_seq_length,    # TODO - add support for individual training
        "select_layers":                args.select_layers,
        "disable_tqdm":                 args.disable_tqdm,
        "distributed":                  args.distributed,
    }

    # Get elastic configs
    configs, eval_elastic_configs = utils.get_elastic_configs(args)

    # Print to command line
    print("-"*20)
    print(f"Elastic configuration has total of {len(configs)} models")
    print(f"Eval configurations:")
    pprint.pp(
        [c.get_name() for c in eval_elastic_configs], 
        compact=True, width=200); 
    print("-"*20)
    print("Script parameters:")
    pprint.pp(train_kwargs, width=200)

    # Save script args both dictionary and command line string
    save_args(args, output_dir)

    ################### Train ###################

    # INDIVIDUAL

    if args.train_type == 'individual':
        # Set data
        train_kwargs = set_train_data(args, train_kwargs, output_dir)

        # Train each config individually, train smallest and largest first
        print("Going to run individual training on eval configs")        
        eval_results={}
        for c in eval_elastic_configs:
            print("-"*20)
            print(f"Training model ({c.get_name()})")            
            indiv_dir = output_dir / c.get_name() 
            os.makedirs(indiv_dir)
            eval_stats={}
            train_kwargs.update({
                'depth':                c.depth,
                'attention_ratio':      c.attention_ratio,
                'intermediate_ratio':   c.intermediate_ratio,
                'hidden_dimension':     c.hidden_dimension,
                'output_dir':           indiv_dir,
                'eval_stats':           eval_stats
            })

            train_single.train_and_eval_model(**train_kwargs)

            eval_results[c.get_name()] = eval_stats[args.basenet]
        for key,val in eval_results.items():
            print(key,val)

    # WEIGHT SHARED

    elif args.train_type == 'ws_sandwich':
        # Add configs for training and eval
        train_kwargs.update({
            'elastic_configs': configs,
            'eval_elastic_configs': eval_elastic_configs,
        })

        for trial in range(args.trials):
            trial_dir = output_dir / str(trial)
            os.makedirs(trial_dir)
            # Set data        
            train_kwargs = set_train_data(args, train_kwargs, trial_dir)
            train_kwargs['output_dir'] = trial_dir

            print(f"Going to run weight shared training - Trial {trial + 1} out of {args.trials}")                    
            ws_train.train_and_eval_model_ws(**train_kwargs)

    # Mark completed
    open(output_dir / 'completed.log', 'wb').close()

import argparse
from pathlib import Path
import ws_train
import pickle 
import torch 
import utils
import pprint

PRINT_WIDTH = 200

def get_best_val_test_acc(all_metrics):
    best_val = 0
    test_acc = 0
    best_idx = 0
    for i, entry in enumerate(all_metrics):
        if entry['val_acc'] > best_val:
            best_val = entry['val_acc']
            test_acc = entry['test_acc']
            best_idx = i
    return best_idx, best_val, test_acc


def print_results(metrics_file, individual=True):
    '''
    Print results for metrics file.
    '''
    if individual:
        keep = ('epoch', ) + \
            ('train_acc', 'train_loss') + \
            ('val_acc', 'val_loss') + \
            ('test_acc', 'test_loss')
    else:
        keep = ('epoch', 'subnet_val_accs', 'subnet_test_accs', 'subnet_train')


    # Load / print metrics
    all_metrics = torch.load(metrics_file)
    if not isinstance(all_metrics, list):
        all_metrics = [all_metrics]

    for epoch_metrics in all_metrics:
        metrics = {k:epoch_metrics[k] for k in keep if k in epoch_metrics}
        pprint.pp(metrics)

    # Print best
    print(
        "Best checkpoint based on val accuracy: " \
        "Epoch Num = {}, Val = {}, Test = {}"
        .format(*get_best_val_test_acc(all_metrics)))


def print_break():
    print('-'*20 + '\n')


def load_exp_args(path):
    exp_args = pickle.load(
        open(path / 'config.pkl', 'rb'))

    # pprint.pp(f"Experiment script args = {exp_args}",  width=PRINT_WIDTH)
    # print_break()
    return exp_args


def load_eval_configurations(exp_args):
    # Get elastic configs
    configs, eval_config = utils.generate_elastic_configs(exp_args)
    return eval_config

def main(args):

    # Get experiment path
    exp_path = Path(args.exp_dir)
    if not exp_path.is_dir():
        exp_path = 'scripts/outputs_run_ws' / exp_path
        if not exp_path.is_dir():
            print("Experiment path does not exist or is not a directory")
            exit()

    # Get experiment args and eval configs
    eval_config = load_eval_configurations(load_exp_args(exp_path))

    print("Eval config:", eval_config)
    smallest = eval_config[0].get_name()
    largest = eval_config[-1].get_name()

    print(f"Configs: Smallest = {smallest}, Largest = {largest}")
    fname = f'{args.metrics_type}.pt'

    if args.training_type == 'individual':
        for config in eval_config:
            name = config.get_name()
            print(name); print_break()
            print_results(exp_path / name / fname); print()
    elif args.training_type == 'ws':
        # Get subpath for trial
        trial_path = exp_path / args.trial
        print_results(trial_path / fname)
    else:
        print("Training type not supported")
        exit()



if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp_dir", type=str, required=True, 
        help="Path to experiment directory")
    parser.add_argument("--training_type", type=str, required=True,
        help="Training type for this experiment. One of 'ws' or 'individual'")
    parser.add_argument("--trial", type=str, default='0',
        help="Name of trial to use")
    parser.add_argument("--indiv_config", type=float, nargs='+',
        help= \
            "For individual only. Select which results to show. " \
            "If not set, print smallest and largest models by default.")
    parser.add_argument("--metrics_type", type=str, default='all_metrics',
        help="Keyword to identify which metrics file to use for results. "+
             "Use either 'all_metrics' or 'epoch_<i>_metrics'")

    args = parser.parse_args()
    main(args)

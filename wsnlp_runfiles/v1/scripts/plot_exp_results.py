import argparse
from pathlib import Path
import ws_train
import pickle 
import torch 
import utils
import pprint

PRINT_WIDTH = 200

def load_metrics(f):
    # keep = ('epoch', 'subnet_val_accs', 'subnet_test_accs', 'subnet_train')
    keep = ('epoch', 'train_acc', 'val_acc', 'test_acc')
    metrics = torch.load(f)
    for i, epoch_metrics in enumerate(metrics):
        # print(epoch_metrics.keys())
        metrics[i] = {k:epoch_metrics[k] for k in keep if k in epoch_metrics}
    return metrics

def print_results(metrics_file, individual=True):
    '''
    Print results for metrics file.
    '''
    metrics = load_metrics(metrics_file)
    for epoch_metrics in metrics:
        pprint.pp(metrics, width=PRINT_WIDTH)

def print_break():
    print('-'*20 + '\n')

def process_exp(exp_path, training_type='ws'):
    exp_args = pickle.load(
        open(exp_path / 'config.pkl', 'rb'))
    print(f"Experiment: {exp_path}:"); 
    print_break()
    pprint.pp(exp_args,  width=PRINT_WIDTH)
    print_break()

    # # Get elastic configs
    # configs = utils.generate_elastic_configs(exp_args)
    # smallest = configs[0].get_name()
    # largest = configs[-1].get_name()
    # print(f"Configurations: Smallest = {smallest}, Largest = {largest}")

    if training_type == 'ws':
        fname = f'{args.metrics_type}.pt'
        return (exp_args, load_metrics(exp_path/fname))


def main(args):
    # Look for metrics files in experiment directory
    exp_paths = [Path(x) for x in args.exp_dirs]
    for i, p in enumerate(exp_paths):
        if not p.is_dir():
            # check standard folder
            newp = Path('scripts/outputs_run_ws') / p
            if not newp.is_dir():
                print(f"Experiment'{p}' not found")
                exit(1)
            exp_paths[i] = newp
    for p in exp_paths:
        exp_args, metrics = process_exp(p, args.training_type)
        # Custom logic
        # print("Learning rate =", args['lr'])
        train_accs = [e['train_acc'] for e in metrics]
        val_accs = [e['val_acc'] for e in metrics]
        test_accs = [e['test_acc'] for e in metrics]
        print(train_accs)
        print(val_accs)
        print(test_accs)




if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp_dirs", nargs='+', required=True, 
        help="Path to experiment directory")
    parser.add_argument("--training_type", type=str, default='ws',
        help="Training type for this experiment. One of 'ws' or 'individual'")
    parser.add_argument("--indiv_config", type=float, nargs='+',
        help= \
            "For individual only. Select which results to show. " \
            "If not set, print smallest and largest models by default.")
    parser.add_argument("--metrics_type", type=str, default='all_metrics',
        help="Keyword to identify which metrics file to use for results. "+
             "Use either 'all_metrics' or 'epoch_<i>_metrics'")

    args = parser.parse_args()
    main(args)

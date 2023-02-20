import torch
from transformers import get_linear_schedule_with_warmup, BertForMaskedLM
from tqdm import tqdm
import numpy as np
import datasets
import wandb
import random
import os
from dataclasses import dataclass, fields

from model import load_model, load_tokenizer, get_subnet, disable_dropout
import data as src_data
import constants
import train_utils
import pprint
from collections import OrderedDict

# Elastic configuration functions. TODO: move to seperate module
def sample_subnets(elastic, num_subnets,is_bucket=False):
    '''
    Return smallest, largest and num_subnets sample (if possible)
    '''

    # make copy of list so as to not modify it with pop
    l = elastic[:]
    subnets = []

    # remove smallest and largest nets
    largest, smallest = l.pop(-1), l.pop(0)
    
    # sample subnets (without replacement)
    if num_subnets > 0:
        if not is_bucket:
            subnets = random.sample(l, num_subnets)
        else:
            subnets=[]
            bucket_size = len(l)//num_subnets
            for i in range(0,len(l),bucket_size):
                subnet = random.sample(l[i:i+bucket_size],1)[0]
                subnets.append(subnet)
            

    return smallest, subnets, largest

@dataclass
class ElasticConfig:
    '''
    Wrapper class for single model elastic configuration 
    '''
    depth: int
    attention_ratio: float
    intermediate_ratio: float
    hidden_dimension: int

    def get_name(self) -> str:
        return f'd_{self.depth}_'\
               f'ar_{self.attention_ratio}_' \
               f'ir_{self.intermediate_ratio}_' \
               f'h_{self.hidden_dimension}'

    @staticmethod
    def from_name(name: str):
        '''
        Convert string of format 'd_<D>_ar_<AR>_ir_<IR>_h_<H>' to config
        '''
        s = name.split('_')
        d, ar, ir, h = [s[i] for i in range(1, len(s), 2)]
        return ElasticConfig(depth=int(d), attention_ratio=float(ar), intermediate_ratio=float(ir), hidden_dimension=int(h))

def generate_elastic_configs(
        elastic_depth, 
        elastic_attention, 
        elastic_intermediate, 
        elastic_hidden, 
        elastic_width=None,
        coupling=False,
        eval_config_percentile=None):
    '''
    Create all valid configurations of given elasticities
    Args:
        depth, attention and intermediate are 
        iterables of valid elastic values for these dimensions
        elastic_width overrides attention and intermediate, making them
        them match each other in configurations
    Returns:
        List[ElasticConfig] of all possible configurations with smallest first and largest last.
    '''
    should_eval = eval_config_percentile is not None
    def get_eval_idx(n1, n2, n3, n4):
        '''
        n<i> is the size of dimension i in the search space
        '''
        eval_idx = {} 
        if should_eval:
            eval_idx = {
                (   int(p * (n1)),
                    int(p * (n2)),
                    int(p * (n3)),
                    int(p * (n4)),
                ) 
                for p in eval_config_percentile
            }
        return eval_idx

    configs = []
    eval_configs=[]

    # Start main function

    # (elastic width) Attention ratio and intermediate ratio \
    # are coupled to be the same value 
    if elastic_width is not None:
        for d in elastic_depth:
            for r in elastic_width:
                for h in elastic_hidden:
                    configs.append(
                        ElasticConfig(
                            depth=d,
                            attention_ratio=r,
                            intermediate_ratio=r,
                            hidden_dimension=h))
        return configs
    
    if not coupling:
        # Get eval indices for each dimension based on percentiles

        eval_idx = get_eval_idx(len(elastic_depth), len(elastic_attention), len(elastic_intermediate), len(elastic_hidden))
        # Loop over dimensions
        for idx_d, d in enumerate(elastic_depth):
            for idx_ar, ar in enumerate(elastic_attention):
                for idx_ir,ir in enumerate(elastic_intermediate):
                    for idx_h, h in enumerate(elastic_hidden):
                        elastic_config=ElasticConfig(
                                depth=d, 
                                attention_ratio=ar, 
                                intermediate_ratio=ir,
                                hidden_dimension=h,)
                        configs.append(elastic_config)


                        # Add to eval if in eval set
                        if should_eval and (idx_d, idx_ar, idx_ir, idx_h) in eval_idx:
                            eval_configs.append(elastic_config)
    else:
        # Couple elastic depth and attention
        max_length = max(len(elastic_depth), len(elastic_attention))
        elastic_depth_step = (len(elastic_depth) - 1) / (max_length - 1)
        elastic_attention_step = (len(elastic_attention) - 1)/(max_length - 1)
        depth_idx = 0
        attention_idx = 0
        eval_idx = get_eval_idx(max_length, max_length, len(elastic_intermediate), len(elastic_hidden))

        # Loop over dimensions
        for i in range(max_length):
            # Couple
            d = elastic_depth[int(i*elastic_depth_step)]
            ar = elastic_attention[int(i*elastic_attention_step)]
            for idx_ir, ir in enumerate(elastic_intermediate):
                for idx_h, h in enumerate(elastic_hidden):
                    # Append config
                    elastic_config=ElasticConfig(
                                depth=d, 
                                attention_ratio=ar, 
                                intermediate_ratio=ir,
                                hidden_dimension=h)
                    configs.append(elastic_config)
                    # Add to eval set
                    if should_eval and(i, i, idx_ir, idx_h) in eval_idx:
                        eval_configs.append(elastic_config)

    if should_eval:
        eval_configs = [configs[0]] + eval_configs + [configs[-1]]

    return configs, eval_configs

################## weight shared training #####################
def train_step(
        model,
        batch,
        optimizer,
        scheduler,
        elastic_configs,
        num_subnets,
        device,
        teacher=None,
        regularize_once=False,
        alpha=None,
        clip_gradient=False,
        task=None,
        nsp=False,
        preserve_finetune_hidden_size=False

    ):
    '''
    Train on one batch and step.
    '''
    soft_labels = None
    smallest_metrics, largest_metrics = dict(), dict()
    subnet_metrics = []
    accs, losses = [], []

    # Get subnet configs to train
    smallest, subnets, largest = sample_subnets(elastic_configs, num_subnets)

    # Use fixed teacher for KD. Only one model at a time on the GPU.
    if teacher:
        model = model.cpu()
        teacher = teacher.to(device)
        outputs = train_utils.apply_to_batch(teacher, batch)
        soft_labels = train_utils.get_soft_labels(outputs, nsp=nsp)
        teacher = teacher.cpu()
    model = model.to(device)


    # Forward pass using sandwich rule. 
    # apply largest model first to get distillation labels
    for config in [largest, smallest] + subnets:

        # Select subnet 
        child = get_subnet(
                model, 
                config.depth, 
                config.attention_ratio, 
                config.intermediate_ratio,
                config.hidden_dimension,
                preserve_finetune_hidden_size
                )



        # Disable dropout for all models except largest
        if regularize_once and config != largest:
            disable_dropout(child)
        else:
            child.train()
                
        # Forward pass and loss
        outputs, loss = train_utils.accumulate_grad(child, batch, soft_labels=soft_labels, alpha=alpha)
        acc = train_utils.get_model_accuracy(outputs, labels=batch['labels'],task=task)

        # Save results
        metrics = {
            'elastic_config':   config,
            'acc':              acc,
            'loss':             loss,
        }
        accs.append(acc); losses.append(loss)

        # Largest - set soft_labels 
        if config == largest:
            largest_metrics = metrics
            # Use largest model outputs for inplace distillation
            if teacher is None:
                soft_labels = train_utils.get_soft_labels(outputs, nsp=nsp)
        # Smallest
        elif config == smallest:
            smallest_metrics = metrics
        else:
            subnet_metrics.append(metrics)

    # Clip gradient
    if clip_gradient:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer step and zero grad
    train_utils.step_optim(optimizer, scheduler)
    return {
        'smallest':     smallest_metrics, # Dict
        'largest':      largest_metrics,  # Dict
        'subnets':      subnet_metrics,   # List[Dict]
        'mean_acc':     np.mean(accs),
        'mean_loss':    np.mean(losses),
    }

def get_train_scores(train_steps):
    '''
    Aggregates train metrics across steps

    Returns:
        Dict[str, float]: Loss and accuracy scores from training
    '''
    log_fields = dict()
    # Get min/maxnet results
    for a in ['smallest', 'largest']:
        for b in ['acc', 'loss']:
            log_fields[a + '_' + b] = np.mean([m[a][b] for m in train_steps])

    # Get mean result across steps and sampled nets
    for k in ['mean_acc', 'mean_loss']:
            log_fields[k] = np.mean([m[k] for m in train_steps])

    return log_fields

# Get metric value for each subnet
def get_scores(metrics , metric_key):
    subnet_scores = []
    for config, subnet_metrics in metrics.items():
        # skip over aggregate values such as mean_acc or mean_loss
        if not isinstance(subnet_metrics, dict):
            continue

        score = subnet_metrics[metric_key]
        subnet_scores.append(score)
        
    return np.asarray(subnet_scores)

def get_val_score(val_metrics, key='acc'):
    '''
    Get validation score. Return mean across subnets in val_metrics.

    The val metrics should be a dictionary mapping config (str) -> metrics (dict)

    If 'loss' is used as a key then the loss will be inverted (L -> 1/L)
    so that higher val score corresponds to a better result.
    '''
    scores = get_scores(val_metrics, key)

    # Compute val score for checkpointing/early stopping
    if key == 'loss':
        scores = 1. / scores

    # Return mean
    return scores.mean()

def eval_model_configs(model, eval_loader, elastic_configs,task,preserve_finetune_hidden_size=False):
    '''

    Eval all given elastic configs on eval loader data

    Returns:
        results: Dict[elastic config] -> metrics dict
    '''

    results = OrderedDict()

    # For each config
    for config in elastic_configs:
        
        # Select subnet 
        child = get_subnet(
            model, 
            config.depth, 
            config.attention_ratio, 
            config.intermediate_ratio,
            config.hidden_dimension,
            preserve_finetune_hidden_size=preserve_finetune_hidden_size
            )



        # Eval
        # Returns dict with keys 'acc', 'loss' -> mean accuracy, mean loss
        metrics = train_utils.eval_model(child, eval_loader, task)
        results[config.get_name()] = metrics

    for key in ['acc', 'loss']:
        results[f'mean_{key}'] = get_scores(results, key).mean()

    # Return 
    return results


def get_eval_metrics(model, val_loader, test_loader, eval_elastic_configs, task,preserve_finetune_hidden_size=False):

    eval_metrics = dict()
    for name, ldr in zip(('val', 'test'), (val_loader, test_loader)):
        if len(ldr) > 0:
            m = eval_model_configs(
                model, ldr,
                elastic_configs=eval_elastic_configs,
                task=task,
                preserve_finetune_hidden_size=preserve_finetune_hidden_size)

            eval_metrics[name] = m

    return eval_metrics

def eval_checkpoint(
        model,
        eval_metrics,
        output_dir,
        weights_dir,
        last_train_steps,
        val_score_key,
        best_val_score,
        epoch_num,
        step_num,
        save_model=False,
        save_best_only=False,
    ):
    '''
    Helper to do eval and checkpoint based on criteria.
    Returns eval metrics and validation score used for checkpointing.
    '''
    avg_train_metrics = get_train_scores(last_train_steps)
    # Save eval metrics
    eval_metrics['epoch_num'] = epoch_num
    eval_metrics['step_num'] =  step_num
    eval_metrics['train_metrics_since_last_eval'] = avg_train_metrics
    torch.save(eval_metrics, output_dir / f'epoch_{epoch_num}_step_{step_num}_metrics.pt')

    # Checkpointing, save models with best val score so far
    val_score = get_val_score(eval_metrics['val'], key=val_score_key)
    if save_model and (val_score > best_val_score or not save_best_only):
        train_utils.save_checkpoint(
            weights_dir, model, epoch_num, step_num, 
            save_best_only=save_best_only) 

    return eval_metrics, avg_train_metrics, val_score

def print_tqdm_results_for_eval(tqdm, eval_metrics, avg_train_metrics):
    # train results
    tqdm.write(
        f"Training scores (since last eval):\n" +\
        f"{pprint.pformat(avg_train_metrics)}\n" +\
        f"-"*20 
    )

    # eval results
    tqdm.write(
        f"Validation scores: \n"+\
        f"{pprint.pformat(eval_metrics['val'])}\n"+\
        f"-"*20 + '\n' +\
        f"Test scores: \n"+\
        f"{pprint.pformat(eval_metrics['test'])}\n"
    )
    print("-"*80)

## the configurations should be defined in the run script.
## the configuration should be in ascending order -> sort otherwise.
def train_and_eval_model_ws(
        model_key, 
        num_epochs, 
        train_dataset,
        val_dataset, 
        test_dataset, 
        output_dir, 
        device, 
        batch_size, 
        learning_rate,
        elastic_configs,
        max_seq_length,        
        eval_elastic_configs=[],        
        save_model=False,
        use_wandb=False, 
        alpha=None,
        num_subnets=None,
        constant_schedule=False,
        linear_schedule_with_warmup=False,
        cosine_schedule_with_warmup=False,
        warmup=None,
        regularize_once=False,
        clip_gradient=False,
        save_best_only=False,
        task=None,
        pretrain_chkpt=None,
        max_train_steps=None,
        eval_every_num_steps=None,
        preserve_finetune_hidden_size=False,
        task_config=None,
        select_layers='',
        disable_tqdm=False,
        distributed=False,
        **kwargs):

    # Metric to use for deciding to checkpoint for save_best_only
    chkpt_metric = 'loss'

    # Setup model save directory
    if save_model:
        weights_dir = output_dir / 'model_weights'
        os.makedirs(weights_dir, exist_ok=True)


    # Load model. 
    print("Loading model and preparing batches...") 
    model = load_model(
        model_key,
        pretrain_chkpt=pretrain_chkpt,
        task=task, 
        task_config=task_config,
        select_layers=select_layers)

    # Load tokenizer
    tokenizer = load_tokenizer(model_key)

    # If pretrain, use a fixed teacher net for distillation
    teacher=None
    if task == 'pretrain':
        teacher = BertForMaskedLM.from_pretrained(model_key)
        teacher.eval()
        num_epochs = 1

    # Handle distributed
    if distributed:
        model = torch.nn.DataParallel(model)
        if teacher:
            teacher = torch.nn.DataParallel(teacher)

    # Create dataloader    
    all_ds = (train_dataset, val_dataset, test_dataset)
    load_func_map = {
        "sst2":     src_data.glue_loader,
        "mnli":     src_data.glue_loader,        
        "pretrain": src_data.pretrain_loader,
    }
    assert task in load_func_map   
    train_loader, val_loader, test_loader = [
        load_func_map[task](tokenizer, ds, batch_size, max_seq_length, task_name=task) 
        if ds is not None else []
        for ds in all_ds]   

    # Training step calculations
    if max_train_steps:
        total_train_steps = max_train_steps
        # Set num_epochs to very large number so that max_train_steps is used
        num_epochs = constants.INF_EPOCHS if task != "pretrain" else 1 
    else:
        total_train_steps = len(train_loader) * num_epochs

    # Optimizer and learning rate scheduler
    optimizer, scheduler = \
        train_utils.get_optimizers(
            model, batch_size,
            total_train_steps, learning_rate,
            constant_schedule=constant_schedule,
            linear_schedule_with_warmup=linear_schedule_with_warmup,
            cosine_schedule_with_warmup=cosine_schedule_with_warmup,
            warmup=warmup,)

    # Train and eval setup
    best_val_score = 0
    completed_steps = 0
    all_eval_metrics = [] 
    train_step_metrics = [] # contains train metrics per step/batch
    print("-"*20)
    print("Starting training")
    print(f"""
        Batch size = {batch_size}
        Max epochs = {num_epochs}, 
        Max train steps = {max_train_steps}, 
        Total train steps = {total_train_steps},
        Eval every steps = {eval_every_num_steps},
    """)
    print("-"*80)
    # Train and eval
    for epoch in range(num_epochs):
        tqdm_loader = tqdm(train_loader, unit="batch", disable=disable_tqdm)
        if not max_train_steps:
            tqdm_loader.set_description(f"Epoch {epoch + 1} of {num_epochs}")

        # Train
        model.train()
        for step, batch in enumerate(tqdm_loader):
            # Step
            train_metrics = train_step(
                model, batch, optimizer, scheduler, 
                elastic_configs, num_subnets, device,
                teacher=teacher,
                regularize_once=regularize_once, 
                alpha=alpha, 
                clip_gradient=clip_gradient, 
                task=task,
                preserve_finetune_hidden_size=preserve_finetune_hidden_size)


            # Save and print train step results
            train_step_metrics.append(train_metrics)
            mean_acc, mean_loss = [round(train_metrics[k], 4) for k in ('mean_acc', 'mean_loss')]
            tqdm_loader.set_description(f'[Batch Acc] {mean_acc} | [Batch Loss] {mean_loss}')

            completed_steps += 1


            # Exit early if max train steps is hit OR end of epoch
            if (max_train_steps is not None and completed_steps >= max_train_steps) \
            or (step  == len(tqdm_loader) - 1):
                break

            # Eval - if eval_every_num_steps
            if eval_every_num_steps and completed_steps % eval_every_num_steps == 0:                    
                # Eval - every epoch or end of training
                tqdm_loader.write(f"Running eval... [Epoch {epoch + 1} | Step {step + 1}/{len(tqdm_loader)}]" + '\n' + '-'*20)
                steps_since_eval = eval_every_num_steps
                eval_metrics = get_eval_metrics(
                    model.to(device), 
                    val_loader, 
                    test_loader, 
                    eval_elastic_configs, 
                    task,
                    preserve_finetune_hidden_size=preserve_finetune_hidden_size
                )                
                eval_metrics, avg_train_metrics, val_score = eval_checkpoint(
                    model,
                    eval_metrics=eval_metrics,
                    output_dir=output_dir, 
                    weights_dir=weights_dir,     
                    last_train_steps=train_step_metrics[-steps_since_eval:],
                    val_score_key=chkpt_metric,
                    best_val_score=best_val_score,
                    epoch_num=epoch + 1,
                    step_num=completed_steps, 
                    save_model=save_model,
                    save_best_only=save_best_only,
                )

                all_eval_metrics.append(eval_metrics)
                best_val_score = max(val_score, best_val_score)

                # Print results
                print_tqdm_results_for_eval(tqdm_loader, eval_metrics, avg_train_metrics)

                # Log to wandb
                if use_wandb:
                    wandb.log({
                        "Train Loss": mean_acc,
                        "Train Acc": mean_loss,
                        "Eval Acc": get_val_score(eval_metrics['test']),
                        "Step" : completed_steps})

                    # log individual subnet results
                    for name, metric in eval_metrics['test'].items():
                        wandb.log({name+"_acc" : metric["acc"],
                            subnet_key+"_loss" : metric["loss"],
                            "Step" : completed_steps                            
                            })

        # Eval - every epoch or end of training

        tqdm_loader.write(f"Running eval... [Epoch {epoch + 1} | Step {step + 1}/{len(tqdm_loader)}]" + '\n' + '-'*80)
        steps_since_eval = len(train_step_metrics) // (epoch + 1)
        eval_metrics = get_eval_metrics(
            model.to(device), 
            val_loader, 
            test_loader, 
            eval_elastic_configs, 
            task,
            preserve_finetune_hidden_size=preserve_finetune_hidden_size

        )                
        eval_metrics, avg_train_metrics, val_score = eval_checkpoint(
            model,
            eval_metrics=eval_metrics,
            output_dir=output_dir, 
            weights_dir=weights_dir,     
            last_train_steps=train_step_metrics[-steps_since_eval:],
            val_score_key=chkpt_metric,
            best_val_score=best_val_score,
            epoch_num=epoch + 1,
            step_num=completed_steps, 
            save_model=save_model,
            save_best_only=save_best_only,
        )

        all_eval_metrics.append(eval_metrics)
        best_val_score = max(val_score, best_val_score)

        # Print results
        print_tqdm_results_for_eval(tqdm_loader, eval_metrics, avg_train_metrics)

        # Log to wandb
        if use_wandb:
            wandb.log({
                "Train Loss": mean_acc,
                "Train Acc": mean_loss,
                "Eval Acc": get_val_score(eval_metrics['test']),
                "epoch" : epoch + 1})

            # log individual subnet results
            for name, metric in eval_metrics['test'].items():
                wandb.log({name+"_acc" : metric["acc"],
                    subnet_key+"_loss" : metric["loss"],
                    "epoch" : epoch + 1                            
                    })

        # Have to exit early again here since break only exits out of the inner loop
        if max_train_steps is not None and completed_steps >= max_train_steps:
            break

    # Save final metrics and model
    train_utils.save_results(output_dir, all_eval_metrics, train_step_metrics)
    if save_model:
        train_utils.save_checkpoint(weights_dir, model, epoch + 1, completed_steps, final=True)

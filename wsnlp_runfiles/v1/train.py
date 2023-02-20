##########
# For non weight shared model training
##########

# TODO - update to handle new load_model and load_tokenizer functions
# from model import load_model_and_tokenizer
import data
import torch
from tqdm import tqdm
import numpy as np
import datasets
import train_utils
import os

def train(
        model, 
        train_loader, 
        optimizer, 
        scheduler, 
        epoch,
        clip_gradient=False,
        task=None):

    # Setup
    model.train()
    train_loss = []
    train_acc = []
    tepoch = tqdm(train_loader, unit="batch")
    tepoch.set_description(f"Epoch {epoch}")
    # Load batches and update
    for sample in tepoch:
        sample = train_utils.batch_col_to_dict(sample, task)

        # Forward pass and loss
        outputs, loss = train_utils.accumulate_grad(model, sample)
        if clip_gradient:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        train_utils.step_optim(optimizer, scheduler)

        # Compute accuracy
        acc = train_utils.get_model_accuracy(outputs, labels=sample['labels'],task=task)

        # Print and store batch results
        train_utils.update_batch_results(acc, loss, train_acc, train_loss, tepoch)

    return tepoch, train_acc, train_loss,

def eval(model, test_loader,task=None):
    # Setup
    return train_utils.eval_model(model, test_loader,task=task)

def train_and_eval_model(
        model_key,
        num_epochs, 
        train_dataset, 
        val_dataset,
        test_dataset, 
        output_dir, 
        device, 
        batch_size,
        learning_rate,
        save_model=False,
        use_wandb=False, 
        attention_ratio=1.0, 
        intermediate_ratio=1.0, 
        depth=-1,
        hidden_dimension=-1,
        constant_schedule=False,
        linear_schedule_with_warmup=False,
        cosine_schedule_with_warmup=False,
        warmup=0.1,
        clip_gradient=True,
        eval_stats={},
        pretrain_chkpt=None,
        task=None,
        max_seq_length=512,
        **kwargs,):

    if save_model:
        weights_dir = output_dir / 'model_weights'
        os.makedirs(weights_dir, exist_ok=True)

    # Load model, data, optimizer and learning rate schedule
    print("Loading model and preparing batches...") 
    model, tokenizer = load_model_and_tokenizer(
        model_key, device, 
        attention_ratio=attention_ratio, 
        intermediate_ratio=intermediate_ratio, 
        depth=depth,
        pretrain_chkpt=pretrain_chkpt,
        task=task,
        hidden_dimension=hidden_dimension)

    # Dataloader, optimizer and learning rate scheduler
    train_loader, val_loader, test_loader = [
        data.sst_loader(tokenizer, ds, batch_size,max_seq_length) 
        for ds in (train_dataset, val_dataset, test_dataset)]
    optimizer, scheduler = \
        train_utils.get_optimizers(
            model, batch_size, 
            len(train_loader) * num_epochs, learning_rate,
            constant_schedule=constant_schedule,
            linear_schedule_with_warmup=linear_schedule_with_warmup,
            cosine_schedule_with_warmup=cosine_schedule_with_warmup,
            warmup=warmup)

    # Train and eval
    best_acc = 0
    best_test_acc=0
    all_metrics =[]
    for epoch in range(num_epochs):
        print('-'*20); print(f"Epoch {epoch} of {num_epochs}:")        
        # Train
        tepoch, train_acc, train_loss = train(
            model, train_loader, 
            optimizer, scheduler, 
            epoch, clip_gradient=clip_gradient,
            task=task)

        # Eval every epoch
        val_eval = eval(model, val_loader,task=task)
        val_acc = val_eval['acc']
        test_eval = eval(model, test_loader,task=task)
        test_acc = test_eval['acc']

        # Save and print epoch_metrics
        epoch_metrics = {
            "epoch":        epoch,
            "train_acc":    np.mean(train_acc), 
            "train_loss":   np.mean(train_loss), 
            "val_acc":      val_acc,
            "test_acc":     test_acc,
        }
        all_metrics.append(epoch_metrics)
        print(epoch_metrics)
        train_utils.save_epoch_metrics(output_dir, epoch, epoch_metrics, tepoch)

        # Checkpointing
        if save_model and val_acc > best_acc:
            train_utils.save_checkpoint(weights_dir, model, epoch)
            best_test_acc = test_acc
        best_acc = max(val_acc, best_acc)

        print()
    eval_stats[model_key] = (best_acc,best_test_acc)
    # Save final metrics and model
    train_utils.save_results(output_dir, model, all_metrics)
    if save_model:
        train_utils.save_checkpoint(weights_dir, model, num_epochs, final=True)


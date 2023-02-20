import numpy as np
import datasets
import torch
import torch.nn.functional as F
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
)

acc_metric = datasets.load_metric("accuracy")


def apply_to_batch(model, batch):
    """
    Common to apply bert model to batch and get outputs.
    """
    device = model.device
    batch_on_device = {}
    for key, val in batch.items():
        batch_on_device[key] = val.to(device, non_blocking=True)
    outputs = model(**batch_on_device)
    return outputs


def accumulate_grad(model, batch, soft_labels=None, temperature=1.0, alpha=1.0):
    """
    Common to apply model during train step.
    Accumulate gradient by getting model output, and call loss.backward()

    Use soft_labels when using distillation

    Returns:
        outputs (Tensor):   Results from applying model to batch
        loss (float):       Loss value for batch
    """
    outputs = apply_to_batch(model, batch)
    loss = 0
    if soft_labels is not None:
        # calculate loss
        for logit_type, value in soft_labels.items():
            student = F.softmax(outputs[logit_type] / temperature, dim=1)

            teacher = F.softmax((value / temperature), dim=1)

            kd_loss = -(teacher * student.log()).sum(dim=1).mean()
            student_loss = outputs.loss
            loss += alpha * kd_loss + (1 - alpha) * student_loss

        # ## calculate nsp loss
        # student = F.softmax(
        #     outputs['seq_relationship_logits']/temperature,
        #     dim=1)

        # teacher = F.softmax(
        #     (soft_labels[1]/temperature),
        #     dim=1)

        # kd_loss = -(teacher * student.log()).sum(dim=1).mean()
        # student_loss = outputs.loss
        # loss += alpha * kd_loss + (1 - alpha) * student_loss
    else:
        loss = outputs.loss

    loss.backward()
    return outputs, loss.item()


def step_optim(optimizer, scheduler):
    """
    Common to step optimizer during training loop
    """
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def get_soft_labels(model_outputs, nsp=True):
    soft_labels = {}
    for key in model_outputs:
        if "logits" in key:
            if key == "seq_relationship_logits" and nsp == False:
                continue
            else:
                soft_labels[key] = model_outputs[key].detach()
    return soft_labels


def get_model_accuracy(model_outputs, labels):
    """
    Get accuracy from model logits and labels
    """
    logits = model_outputs["logits"]
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=-1)
    r = acc_metric.compute(predictions=predictions, references=labels.cpu())["accuracy"]
    return r


def eval_model(model, test_loader, task):
    """
    Eval model on test loader batches and return mean accuracy and loss
    """
    model.eval()
    accs = []
    losses = []
    with torch.no_grad():
        for sample in test_loader:
            outputs = apply_to_batch(model, sample)
            acc = get_model_accuracy(outputs, labels=sample["labels"], task=task)
            accs.append(acc)
            losses.append(outputs.loss.item())

    return {
        "acc": np.mean(accs),
        "loss": np.mean(losses),
    }


def get_optimizers(
    model,
    batch_size,
    train_steps,
    lr,
    constant_schedule=False,
    linear_schedule_with_warmup=False,
    cosine_schedule_with_warmup=False,
    warmup=0.1,
):
    """
    Common for getting optimizers/scheduler.
    """
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)

    # Scheduler
    numt = train_steps
    numw = warmup * numt
    if linear_schedule_with_warmup:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=numw, num_training_steps=numt
        )
    elif cosine_schedule_with_warmup:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=numw, num_training_steps=numt
        )
    elif constant_schedule:
        if numw > 0:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=numw
            )
    else:
        print("ERROR: Learning rate schedule not supported.")
        raise ValueError("No learning rate schedule chosen")

    return optimizer, scheduler


def save_train_step_metrics(metrics, output_dir):
    torch.save(metrics, output_dir / "train_step_metrics.pt")


def save_epoch_metrics(output_dir, epoch_num, metrics, tepoch, print_test=True):
    """
    Common to write epoch metrics and save them during training loop
    """
    # Write epoch metrics
    torch.save(
        metrics,
        output_dir / f"epoch_{epoch_num}_metrics.pt",
    )
    # Print results
    tepoch.write(f"Mean Train Acc: {round(metrics['train_acc'], 4)}")
    tepoch.write(f"Mean Train Loss: {round(metrics['train_loss'], 4)}")
    if "subnet_train_accs" in metrics:
        tepoch.write(f"Subnet Train Accs: {metrics['subnet_train_accs']}")
    if "subnet_train_loss" in metrics:
        tepoch.write(f"Subnet Train Loss: {metrics['subnet_train_loss']}")

    tepoch.write(f"Mean Validation Acc: {round(metrics['val_acc'], 4)}")
    if "subnet_val_accs" in metrics:
        tepoch.write(f"Subnet Val Accs: {metrics['subnet_val_accs']}")

    if print_test:
        tepoch.write(f"Mean Test Acc: {round(metrics['test_acc'], 4)}")
        if "subnet_test_accs" in metrics:
            tepoch.write(f"Subnet Test Accs: {metrics['subnet_test_accs']}")


def update_batch_results(cur_acc, cur_loss, train_acc, train_loss, tepoch):
    """
    Common to update results during batch training and write to tqdm progress bar
    """
    train_acc.append(cur_acc)
    train_loss.append(cur_loss)
    tepoch.set_description(
        f"[Batch Acc] {round(cur_acc, 3)} | " f"[Batch Loss] {round(cur_loss, 3)}"
    )


def save_checkpoint(
    output_dir, model, epoch_num, step_num=None, final=False, save_best_only=False
):
    """
    Common to save model checkpoint
    """
    ts = f"epoch_{epoch_num}_step_{step_num}"
    if save_best_only:
        save_model(model, output_dir / f"{ts}_best_val_weights.pt")
        return
    if final:
        save_model(model, output_dir / f"{ts}_final_weights.pt")
        print("Training completed.")
    else:
        save_model(
            model,
            output_dir / f"{ts}_chkpt_weights.pt",
        )


def save_results(output_dir, all_eval_metrics, train_step_metrics):
    """
    Common that saves final results for experiment.
    """
    torch.save(all_eval_metrics, output_dir / f"all_eval_metrics.pt")

    torch.save(train_step_metrics, output_dir / f"all_train_step_metrics.pt")


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Saved model to {filepath}")

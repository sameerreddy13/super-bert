import math
import sys
from typing import Iterable
import torch

# from timm.data import Mixup
# from timm.utils import accuracy, ModelEma
# from timm.utils.model import unwrap_model
import utils
import random
import time
import train_utils


def sample_configs(choices):
    config = {}
    dimensions = ["mlp_ratio", "num_heads"]
    depth = random.choice(choices["depth"])

    # NOTE -  sample for each layer
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    # NOTE - embedding dim is fixed across net
    config["embed_dim"] = [random.choice(choices["embed_dim"])] * depth

    config["layer_num"] = depth

    return config


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epoch: int,
    teacher_model: torch.nn.Module = None,
    clip_grad: bool = None,
    config_choices: dict = None,
) -> dict:

    model.train()

    # Set random seed
    random.seed(epoch)

    # Setup logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for batch in metric_logger.log_every(dataloader, print_freq, header):
        # Sample random config
        config = sample_configs(choices=config_choices)
        # model.set_sample_config(config)
        # print("Sampled model config: {}".format(config))

        optimizer.zero_grad()

        outputs = train_utils.apply_to_batch(model, batch)
        if teacher_model:
            # TODO - implement distillation
            pass
        else:
            loss = outputs.loss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged Train stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: Iterable,
    config_choices: dict = None,
    metrics_title="Val",
) -> dict:

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Validation:"

    # Sample random config
    config = sample_configs(choices=config_choices)
    # model.set_sample_config(config)
    # print("Sampled model config: {}".format(config))

    # Evaluate on validation set
    for batch in metric_logger.log_every(dataloader, 10, header):
        outputs = train_utils.apply_to_batch(model, batch)
        loss = outputs.loss
        acc1 = train_utils.get_model_accuracy(
            model_outputs=outputs,
            labels=batch["labels"],
        )

        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1, n=len(batch["labels"]))

    print(
        "{} Averaged Stats: Acc {top1.global_avg:.3f} Loss {losses.global_avg:.3f}".format(
            metrics_title, top1=metric_logger.acc1, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

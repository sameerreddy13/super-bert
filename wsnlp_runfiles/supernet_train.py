import datetime
import pdb
import json
import time
from unicodedata import name
import torch
import datasets
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import argparse
import pprint
import pickle
import sys
import yaml

import transformers
from transformers import logging
from transformers.models.bert.modeling_bert_super import (
    SuperBertForSequenceClassification,
)
from transformers.models.bert.modeling_bert import BertForSequenceClassification

# Local modules
import data as src_data
import utils
from config import cfg, update_config_from_file, gen_config
from supernet_engine import train_one_epoch, evaluate

logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(
        "Autoformer style supernet training for GLUE tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--cfg",
        required=True,
        type=str,
        help="Experiment configuration file",
        metavar="CFG_FILE",
    )
    parser.add_argument(
        "--seed", default=0, type=int, metavar="SEED", help="Random seed"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device",
        metavar="DEVICE",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Whether or not to use distributed training",
    )
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    #  Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnli",
        help="GLUE dataset",
        metavar="DATASET",
    )
    parser.add_argument(
        "--limit", type=int, default=-1, help="Limit dataset size", metavar="LIMIT"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Max sequence length",
        metavar="MAX_SEQ_LEN",
    )

    #  Training parameters
    parser.add_argument(
        "--batch-size", default=128, type=int, metavar="BS", help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        metavar="NUM EPOCHS",
        help="Total training epochs",
    )
    # parser.add_argument('--num_subnets', type=int, default=2,
    #                     help="number of subnets to sample for sandwich sampling", metavar='NUM_SUBNETS')
    # parser.add_argument('--kd_weight', type=float, default=1.0, help="

    #  Model parameters
    parser.add_argument(
        "--model",
        default="bert-base-uncased",
        type=str,
        help="Name of model to train",
        metavar="MODEL",
    )
    parser.add_argument(
        "--hidden_dropout",
        type=float,
        default=0.0,
        help="Hidden dropout prob",
        metavar="HIDDEN_DROPOUT",
    )
    parser.add_argument(
        "--attn_dropout",
        type=float,
        default=0.0,
        help="Attn dropout prob",
        metavar="ATTN_DROPOUT",
    )

    # Learning rate parameters
    parser.add_argument(
        "--sched", default="linear", type=str, help="LR scheduler", metavar="SCHEDULER"
    )
    parser.add_argument(
        "--warmup", default=0.1, type=float, help="Warmup proportion", metavar="WARMUP"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate", metavar="LR"
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt", default="adamw", type=str, help="Optimizer", metavar="OPTIMIZER"
    )
    parser.add_argument(
        "--clip_grad", action="store_true", default=True, help="Gradient clipping"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay", metavar="WD"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory",
        metavar="DIR",
    )

    return parser.parse_args()


def save_args(args, output_dir):
    # Save args to output directory
    save_config = vars(args)
    json.dump(save_config, (output_dir / "args.json").open("w"))

    # Save command to output directory
    cmd_elems = sys.argv
    cmd_elems[0] = "python -m scripts.supernet_train"
    cmd_line = " ".join(cmd_elems).strip()
    with (output_dir / "command.txt").open("w") as fp:
        fp.write(cmd_line)


if __name__ == "__main__":
    ########################### SETUP ###########################

    # Parse args
    args = parse_args()
    update_config_from_file(args.cfg)
    print("Args:")
    pprint.pprint(vars(args))

    # Device and seed
    device = torch.device(args.device)
    utils.seed_everything(args.seed)

    # Load text data (and apply args.limit)
    raw_datasets = datasets.load_dataset("glue", args.dataset)

    def limit_ds(ds):
        return ds if args.limit < 0 else ds.select(range(min(args.limit, len(ds))))

    dataset_train = limit_ds(raw_datasets["train"])
    if args.dataset == "mnli":
        dataset_val = {
            "matched": limit_ds(raw_datasets["validation_matched"]),
            "mismatched": limit_ds(raw_datasets["validation_mismatched"]),
        }
    else:
        dataset_val = limit_ds(raw_datasets["validation"])

    tokenizer = transformers.BertTokenizerFast.from_pretrained(args.model)

    def tokenize_dataset(
        dataset,
        dataset_split,
        dataset_name=args.dataset,
        max_seq_length=args.max_seq_length,
    ):
        def tokenize_text(examples):
            """
            Tokenize function
            """
            sentence1_key, sentence2_key = src_data.task_to_keys[dataset_name]
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(
                *args,
                max_length=max_seq_length,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
            )
            return result

        dataset = dataset.map(
            tokenize_text,
            batched=True,
            desc=f"Applying tokenizer to {dataset_split} dataset",
        )
        dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
        dataset = dataset.rename_column("label", "labels")
        return dataset

    # Tokenize datasets
    dataset_train = tokenize_dataset(dataset_train, "train")
    if args.dataset == "mnli":
        dataset_val = {
            "matched": tokenize_dataset(dataset_val["matched"], "val_matched"),
            "mismatched": tokenize_dataset(dataset_val["mismatched"], "val_mismatched"),
        }
    else:
        dataset_val = tokenize_dataset(dataset_val, "val")

    # Create dataloaders
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
    )

    def create_val_dataloader(ds):
        sampler = torch.utils.data.SequentialSampler(ds)
        return sampler, torch.utils.data.DataLoader(
            ds,
            sampler=sampler,
            batch_size=args.batch_size,
            pin_memory=args.pin_mem,
        )

    if args.dataset == "mnli":
        sampler_val_m, dataloader_val_m = create_val_dataloader(dataset_val["matched"])
        sampler_val_mm, dataloader_val_mm = create_val_dataloader(
            dataset_val["mismatched"]
        )
    else:
        sampler_val, dataloader_val = create_val_dataloader(dataset_val)

    if not args.dataset == "stsb":  # for regression
        label_list = dataset_train.features["labels"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # # Load Model. NOTE - Supernet dims set here
    # model = SuperBertForSequenceClassification.from_pretrained(
    #     args.model,
    #     config=transformers.SuperBertConfig(
    #         num_labels=num_labels,
    #         mlp_ratio=cfg.SUPERNET.MLP_RATIO,
    #         embed_dim=cfg.SUPERNET.EMBED_DIM,
    #         num_heads=cfg.SUPERNET.NUM_HEADS,
    #         num_hidden_layers=cfg.SUPERNET.DEPTH,
    #         hidden_dropout_prob=args.hidden_dropout,
    #         attention_probs_dropout_prob=args.attn_dropout,
    #     ),
    # )
    # Load Model. NOTE - Supernet dims set here
    model = BertForSequenceClassification.from_pretrained(
        args.model,
        config=transformers.BertConfig(num_labels=num_labels),
    )
    model.to(device)

    # NOTE - Search space set here
    choices = {
        "num_heads": cfg.SEARCH_SPACE.NUM_HEADS,
        "mlp_ratio": cfg.SEARCH_SPACE.MLP_RATIO,
        "embed_dim": cfg.SEARCH_SPACE.EMBED_DIM,
        "depth": cfg.SEARCH_SPACE.DEPTH,
    }
    print("-" * 40)
    print("Search Space:")
    pprint.pprint(choices)

    # Optimizer
    if args.opt == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")

    # Scheduler
    num_training_steps = len(dataloader_train) * args.epochs
    scheduler = transformers.get_scheduler(
        name=args.sched,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_training_steps * args.warmup,
    )

    # Output dir. Save script args, script command, and cfg yaml
    output_dir = Path(args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        gen_config(output_dir / "cfg.yaml")
        save_args(args, output_dir)

    ########################### TRAIN ###########################
    print("-" * 40)
    print("Start training")
    print(
        f"""
        Batch size = {args.batch_size}
        Max epochs = {args.epochs},
        Total train steps = {num_training_steps},
    """
    )
    print("-" * 40)

    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch} / {args.epochs}")
        # Train one epoch
        train_stats = train_one_epoch(
            model=model,
            dataloader=dataloader_train,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            teacher_model=None,
            clip_grad=args.clip_grad,
            config_choices=choices,
        )

        # Evaluate
        if args.dataset == "mnli":
            val_stats_matched = evaluate(
                model=model,
                dataloader=dataloader_val_m,
                config_choices=choices,
                metrics_title="Val Matched",
            )
            val_stats_mismatched = evaluate(
                model=model,
                dataloader=dataloader_val_mm,
                config_choices=choices,
                metrics_title="Val Mismatched",
            )
            val_acc = (val_stats_matched["acc1"] + val_stats_mismatched["acc1"]) / 2

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_m_{k}": v for k, v in val_stats_matched.items()},
                **{f"val_mm_{k}": v for k, v in val_stats_mismatched.items()},
                "epoch": epoch,
            }
        else:
            val_stats = evaluate(
                model=model,
                dataloader=dataloader_val,
                config_choices=choices,
            )
            val_acc = val_stats["acc1"]

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
            }

        # Save model if val accuracy increased
        if args.output_dir and val_acc > max_accuracy:
            model.save_pretrained(output_dir / "checkpoint")
            # TODO - save optimizer and scheduler to the same directory. Also save epoch and args.

        # Write to log file
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

        max_accuracy = max(max_accuracy, val_acc)
        print(f"\nMax validation accuracy: {max_accuracy: .4f}")
        print(f"Time taken: {time.time() - start_time: .1f} seconds")
        print("-" * 40)

    # Training done
    open(output_dir / "FINISHED.log", "wb").close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("-" * 40)
    print("TRAINING FINISHED. Training time {}".format(total_time_str))

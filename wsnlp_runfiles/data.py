import torch
from transformers import DataCollatorForLanguageModeling
from transformers import AutoConfig
import datasets

"""
Code for fetching and formatting glue data heavily influenced by huggingface run_glue.py script
"""
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def fetch_glue_train_val_test(model_name, task_name, lim, val_size):
    """
    Gets the corresponding task data from the GLUE benchmark.
    Uses the huggingface datasets library to fetch and cache data.

    Formats into train, val, test sets and loads config based on model and task.

    Args:
        model_name: base model name e.g. bert-base-uncased
        task_name: task name, should map to options from datasets.load_dataset("glue", <task_name>) e.g. mnli
        lim: limit on total number of samples returned
        val_size: val split size from train data e.g. 0.2

    Returns:
        train_dataset: Train data
        val_dataset: Val data
        test_dataset: Test data. For MNLI there are two test sets for matched/mismatched. A dictionary of datasets is returned instead.
        idx_dict: Dict for train/val indices. Used for reproducibility.
    """
    # Load/fetch task data from huggingface
    raw_datasets = datasets.load_dataset("glue", task_name)

    # Get train data
    train_dataset = (
        raw_datasets["train"] if lim < 0 else raw_datasets["train"].select(range(lim))
    )

    # Get num labels
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = train_dataset.features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Config
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task=task_name,
    )

    # Use validation set as hold out test set
    if task_name == "mnli":
        test_dataset = {
            "validation_m": raw_datasets["validation_matched"],
            "validation_mm": raw_datasets["validation_mismatched"],
        }
        for k in test_dataset:
            test_dataset[k] = (
                test_dataset[k] if lim < 0 else test_dataset[k].select(range(lim))
            )
    else:
        test_dataset = raw_datasets["validation"]
        test_dataset = test_dataset if lim < 0 else test_dataset.select(range(lim))

    # Split train for train/val sets
    val_dataset = None
    idx_dict = {}
    if val_size > 0.0:
        ds_dict = train_dataset.train_test_split(val_size, shuffle=True)
        train_dataset, val_dataset = ds_dict["train"], ds_dict["test"]
        if "idx" in train_dataset:
            idx_dict = {
                "train_idx": train_dataset["idx"],
                "val_idx": val_dataset["idx"],
            }

    return train_dataset, val_dataset, test_dataset, idx_dict, config


def fetch_pretrain_train_val_test(pretrain_tokenized_path, lim, val_size):
    """
    Get pretrain data along with train, val, test splits.
    These should already be tokenized
    """
    # Load from disk
    ds = datasets.load_from_disk(pretrain_tokenized_path)["train"]
    ds = ds if lim < 0 else ds.select(range(lim))

    # Split train val. No shuffle because shuffling this massive dataset is too slow.
    ds_dict = ds.train_test_split(val_size, shuffle=False)
    train_dataset, val_dataset = ds_dict["train"], ds_dict["test"]

    return train_dataset, val_dataset, None, None


def glue_loader(tokenizer, dataset, batch_size, max_seq_length, task_name):
    """
    Form text dataset into tokenized batches for GLUE task finetuning
    """

    def preprocess_function(examples):
        # Tokenize the texts
        sentence1_key, sentence2_key = task_to_keys[task_name]
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
            truncation=True
        )

        return result

    # Combine 'validation_matched and 'validation_mismatched' for MNLI test.
    # This is temporary, we will want to report both scores separately later.
    # if task_name == 'mnli' and isinstance(dataset, dict):
    # dataset = datasets.interleave_datasets([dataset['validation_m'], dataset['validation_mm']])

    # Tokenize dataset
    dataset = dataset.map(
        preprocess_function, batched=True, desc="Applying tokenizer to dataset"
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
    )
    dataset = dataset.rename_column("label", "labels")

    # Return dataloader
    return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)


def pretrain_loader(tokenizer, dataset, batch_size, max_seq_length, task_name=None):
    """
    Form tokenized dataset into batches for MLM, create data loader
    """
    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    return torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )

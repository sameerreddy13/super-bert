from pathlib import Path
from model import get_tokenizer
from accelerate import Accelerator, DistributedType
import argparse 
import utils
import datasets
from itertools import chain 
import os
import data as src_data

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch and preprocess book corpus data for pretraining", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_dir', type=str, default='bookcorpus_tokenized/',
        help="Save location for preprocessed dataset")

    parser.add_argument('--basenet',type=str, default='bert-large-uncased', 
        help="The pretrained model key (for tokenizer)")

    parser.add_argument('--limit', type=int, default=-1, 
        help="Limit the number of training samples used")

    parser.add_argument('--max_seq_length', type=int, 
        help='Max sequence length for model. Default is to use max allowed by model tokenizer')

    parser.add_argument('--overwrite', action='store_true',
        help='Overwrite save_dir if it exists')

    return parser.parse_args()

def main():
    args = parse_args()
    # Config
    num_proc = None # num processes for workers

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Create output directory(s)
    if args.overwrite:
        utils.overwrite_dir(args.save_dir)
    else:
        os.makedirs(args.save_dir)
    print("Created output directory:", Path(args.save_dir).resolve())

    # Tokenizer
    tokenizer = get_tokenizer(args.basenet)

    # Data
    raw_dataset = datasets.load_dataset("bookcorpus")
    column_names = raw_dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    if args.limit > 0:
        raw_dataset = raw_dataset['train'].select(range(args.limit))
    print("Going to tokenize:", raw_dataset)

    # Get max seq length
    max_seq_length = args.max_seq_length if args.max_seq_length else tokenizer.model_max_length

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    with accelerator.main_process_first():
        tokenized_datasets = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=column_names,
            desc="Running tokenizer on every text in dataset",
        )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )

    # Save tokenized dataset
    tokenized_datasets.save_to_disk(args.save_dir)
    print("Saved tokenized data to directory:", Path(args.save_dir).resolve())
    print("In mem:", tokenized_datasets)
    print("Disk:", datasets.load_from_disk(args.save_dir))


if __name__ == '__main__':
    main()



# `run_perf.py` Summary and Usage

## Purpose
This script is used to train weight-shared models

## Training Arguments
### Training Configuration
-  `--gpu` : specify the id of the GPU being used [ *required*]
-  `--layer` : use this boolean flag to run individual models with varying layer number. The models run here are from pretrained miniature models from hugging face. They have a fixed attention head number and hidden dimension size. The number of layers is not customizable.
-  `--hidden` : use this boolean flag to run individual models with varying hidden dimension size. The number of attention heads is scaled according to the hidden dimension size, but the number of layers is kept at 12.
-  `--miniature` : use this boolean flag to run pretrained BERT miniatures.
-  `--all` : use this boolean flag to run `--layer` , `--hidden`, and `--miniature` runs.
-  `--attention_config` : list argument to define the attention ratio of the model to be run. ex) `--attention_config 0.2 0.4 0.6 0.8 1.0`
-  `--attention_approach` : define whether the head size or the head number is changed. select from `[ "head_num", "head_size_1", "head_size_2"]`
-  `--layer_config` : list argument to define the layer number of the model to be run ex) `--layer_config 2 4 8 12`
-  `--base_model` : the model to which the `attention_config` and `layer_config` would be applied. It is `bert-large-uncased` by default.

*Note that if both `layer_config` and `attention_config` are set, the number of elements in each list should match. and the `i`th layer configuration would be paired with `i`th attention configuration in training*

### Training Hyperparameters


### Others
- `--batch_size` : the batch size of the trained model. 128 by default
- `--id` : the folder to which the training results and models would be saved. By default `main`
- `trials` : the number of repeated trials for each model. 3 by default
- `steps` : the number of steps to run each model for.

*Note that the ramp up ratio for the latency measurement is 0.1. This is defined in `constants.py`*

Example :
	`python run_ws.py --id test_run --gpu 2 --attention_config 0.2 0.4 0.6 0.8 1.0 --layer_config 2 4 8 16 24`

### Directories
- `BASE_DIR` : `./outputs_run_perf`
- files -> `BASE_DIR/id`


### Saved Files
`latency.py` defines file names for each run type.

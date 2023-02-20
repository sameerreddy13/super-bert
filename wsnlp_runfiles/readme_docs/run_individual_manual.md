# `run_individual.py` Summary and Usage

## Purpose
This script is used to train individual models and measure the accuracy

## Training Arguments
### Training Configuration
- `--gpu` : specify the id of the GPU being used [ *required*]
- `--layer` : use this boolean flag to run individual models with varying layer number. The models run here are from pretrained miniature models from hugging face. They have a fixed attention head number and hidden dimension size. The number of layers is not customizable. 
- `--hidden` : use this boolean flag to run individual models with varying hidden dimension size. The number of attention heads is scaled according to the hidden dimension size, but the number of layers is kept at 12.
- `--miniature` : use this boolean flag to run pretrained BERT miniatures.
- `--all` : use this boolean flag to run `--layer` , `--hidden`, and `--miniature`  runs.
-  `--attention_config` : list argument to define the attention ratio of the individual models to be run. ex) `--attention_config 0.2 0.4 0.6 0.8 1.0`
- `--attention_approach` : define whether the head size or the head number is changed. select from `[ "head_num", "head_size_1", "head_size_2"]`
- `--layer_config` : list argument to define the layer number of the individual models to be run ex) `--layer_config 2 4 8 12`
- `--base_model` : the model to which the `attention_config` and `layer_config` would be applied. It is `bert-large-uncased` by default.
*Note that if both `layer_config` and `attention_config` are set, the number of elements in each list should match. and the `i`th layer configuration would be paired with `i`th attention configuration in training*

### Training Hyperparameters
- `--batch_size` : the batch size of the trained model. 64 by default
- `--epochs` : the number of epochs to be run. The default number is define as `EPOCHS` in `constants.py`

### Others
- `--id` : the folder to which the training results and models would be saved. By default `main`
- `--wandb` : use this boolean flag to enable wandb 
- `--save_model` : use this boolean flag to disable saving the trained model


Example :
	`python run_individual.py --id test_run --gpu 2 --attention_config 0.2 0.4 0.6 0.8 1.0 --layer_config 2 4 8 16 24`

### Directories
- `BASE_DIR` : `./outputs_run_individual`
- for `--hidden` : `BASE_DIR/id/vary_hidden_dim`
- for `--layer` : `BASE_DIR/id/vary_layer`
- for `--miniature` : `BASE_DIR/id/vary_miniature`
- for `--attention_config` : `BASE_DIR/id/vary_attention_{attention_approach}`
- for `--layer_config` : `BASE_DIR/id/vary_layer_elastic`
- for both `--attention_config` and `--layer_config` : `BASE_DIR/id/joint`

### Saved Files
- Model weight checkpoints: `basename + '_model_chkpt_weights.pt'`
- Final model weight : `base_name + "_model_final_weights.pt"`
- Accuracy stats: `base_name + "_test_acc.pt`

# `run_ws.py` Summary and Usage

## Purpose
This script is used to train weight-shared models

## Training Arguments
### Training Configuration
- `--gpu` : specify the id of the GPU being used [ *required*]
-  `--elastic_attention_config` : list argument to define the elastic attention ratio dimensions ex) `--attention_config 0.2 0.4 0.6 0.8 1.0`
- `--attention_approach` : define whether the head size or the head number is changed. select from `[ "head_num", "head_size_1", "head_size_2"]`
- `--elastic_layer_config` : list argument to define the elastic layer number dimensions  ex) `--layer_config 2 4 8 12`
- `--base_model` : the model to which the `attention_config` and `layer_config` would be applied. It is `bert-large-uncased` by default.

### Training Hyperparameters
- `--batch_size` : the batch size of the trained model. 64 by default
- `--epochs` : the number of epochs to be run. The default number is define as `EPOCHS` in `constants.py`

### Others
- `--id` : the folder to which the training results and models would be saved. By default `main`
- `--wandb` : use this boolean flag to enable wandb 
- `--save_model` : use this boolean flag to disable saving the trained model


Example :
	`python run_ws.py --id test_run --gpu 2 --attention_config 0.2 0.4 0.6 0.8 1.0 --layer_config 2 4 8 16 24`

### Directories
- `BASE_DIR` : `./outputs_run_ws`
- files -> `BASE_DIR/id`


### Saved Files
- Model weight checkpoints: `"run" + train_config_tag + '_model_chkpt_weights.pt'`
- Final model weight : `"run" + train_config_tag + "_model_final_weights.pt"`
- Accuracy stats: `"run" + train_config_tag + "_test_acc.pt`

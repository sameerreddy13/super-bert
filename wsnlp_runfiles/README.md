# OFA Transformers Runfiles
Runfiles for ofa_transformer experiments

## Setup
- You should have a directory structure where `ofa_transformers_runfiles` and [`ofa_transformers`](https://github.com/irenelee5645/ofa_transformers) are seperate directories under one main directory. 
- Current dependencies require Python 3.8 - 3.10.
 
### Miniconda/Python setup
- Get 3.8 miniconda installer with `wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh`
- Follow [installation](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)

### Install packages
- `cd ofa_transformers_runfiles` 
- Install packages with `pip3 install -r requirements.txt`.

### Install transformers

- `cd ofa_transformers` and run `pip install -e .`.

## Add dependencies 
- If you need to add new dependencies include the new package and version in `requirements.txt`.

## Run scripts

## Structure

- `notebooks/` contains notebooks for experiment evaluation and exploration
- `runscripts/` contains scripts to run experiments and store data.

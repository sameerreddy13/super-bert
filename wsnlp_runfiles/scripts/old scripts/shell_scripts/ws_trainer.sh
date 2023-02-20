#### WEIGHT SHARED ####
## for finetuning from pretrained checkpoint

WS_EXP="test"
WS_CMD="
python -m scripts.run_ws 
    --output_dir                $WS_EXP
    --train_type                ws_sandwich 
    --regularize_once
    --alpha                     1.0
    --num_subnets               8
    --batch_size                64
    --lr                        1e-5
    --constant_schedule
    --warmup                    0.05
    --val_split                 0.05  
    --save_best_only        
    --basenet                   bert-large-uncased
    --task                      sst2
    --coupling
    --select_layers             everyother
    --preserve_finetune_hidden_size
"

## for pretraining a model
# WS_CMD="
# python -m scripts.run_ws 
#     --output_dir                $WS_EXP
#     --ws_sandwich 
#     --regularize_once
#     --alpha                     1.0
#     --num_subnets               2
#     --batch_size                8
#     --lr                        1e-4
#     --linear_schedule_with_warmup
#     --max_train_steps           10000
#     --warmup                    0.1
#     --eval_every_num_steps      1000
#     --val_split                 0.0005  
#     --max_seq_length            512 
#     --save_best_only        
#     --basenet                   bert-large-uncased
#     --task                      sst2
#     --coupling
#     --pretrain_chkpt            ws_pretrain_2
# "


#### INDIVIDUAL ####
INDIV_EXP="indiv_finetune_ws_pretrain"
INDIV_CMD="
python -m scripts.run_ws \
    --output_dir                    $INDIV_EXP
    --basenet                       bert-large-uncased
    --individual
    --elastic_intermediate          0.5 0.6 0.7 0.8 0.9 1.0
    --elastic_attention             0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    --elastic_layer                 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    --batch_size                    128
    --epochs                        8
    --lr                            1e-5
    --linear_schedule_with_warmup
    --warmup                        0.1
    --clip_gradient
    --val_split                     0.05
    --task                          sst2
    --pretrain_chkpt                pretrain_chkpt
"

#### RUN ####
handle_exit() {
    echo """
    Helper for executing run_ws.py script.
    All arguments are required.

    Script usage:
        ws_trainer.sh <train_type> <run_type> [additional args...]

    Args:
        train_type (string): Either use 'ws' or 'individual'.

        run_type (string): Either use 'nohup' or 'interactive'.
        Nohup uses nohup to prevent SIGHUP and runs in background, writing to a log.
        Interactive runs it as normal in the shell.

        additional args (string): Add additional arguments to the main train python script. 
        The string is appended so supply in format such as '--arg1 val1 --arg2 val2...'.
        For example, to use a custom gpu rather than the default (which uses 'cuda' to assign) you can append
        '--gpu 1' to use gpu 1.
    """     
    exit 1
}
# Parse train type
if [[ "$1" == "ws" ]]; then
    CMD=$WS_CMD
    EXP=$WS_EXP
elif [[ "$1" == "individual" ]]; then
    CMD=$INDIV_CMD
    EXP=$INDIV_EXP
else
    echo "<train_type> = $1 not supported"
    handle_exit
fi
# Parse run type, run command
NOHUP=false
if [[ "$2" == "nohup" ]]; then
    echo "Running with nohup and saving stdout to $EXP.log"
    NOHUP=true
elif [[ "$2" == "interactive" ]]; then
    echo "Running interactive"
else
    echo "<run_type> = $2 not supported"
    handle_exit
fi
shift 2

# Echo chosen command w/ gpu and additional args appended
CMD="$CMD $@"
echo "Command:"
echo $CMD
# Run command!
if [[ "$NOHUP" = true ]]; then
    LOG_FILE="$EXP.log"
    mkdir -p "$(dirname $LOG_FILE)"
    eval 'nohup $CMD > logs/$LOG_FILE 2>&1 &'
else
    eval $CMD
fi

    


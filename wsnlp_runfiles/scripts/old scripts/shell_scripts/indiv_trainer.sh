EXP="test"
CMD="
python -m scripts.run_individual \
	--gpu 5
	--save_dir $EXP
	--linear_schedule_with_warmup
	--clip_gradient
    --miniature
"


#### RUN ####
if [[ "$#" -ne 1 ]]; then
	echo """
		Script usage:
			indiv_trainer.sh <run_type> 

		Args:
			run_type (string): Either use 'nohup' or 'interactive'.
			Nohup uses nohup to prevent SIGHUP and runs in background, writing to a log.
			Interactive runs it as normal in the shell.

	"""		
	exit 1
fi

# Echo chosen command
echo "Command:"
echo $CMD


# Parse run type, run command
if [[ "$1" == "nohup" ]]; then
	echo "Running with nohup and saving stdout to $EXP.log"
	eval 'nohup $CMD > logs/$EXP.log 2>&1 &'
elif [[ "$1" == "interactive" ]]; then
	echo "Running interactive"
	eval $CMD
else
	echo "<run_type> = $1 not supported"
	exit 1
fi
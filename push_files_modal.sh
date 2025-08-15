#!/bin/bash
echo "pushing trl files to modal volume"

echo "before" 
modal volume ls RLHF 

modal volume put -f RLHF ./rm_training_corrected.py rm_training_corrected.py
modal volume put -f RLHF ./reward_bench_eval.py reward_bench_eval.py 
modal volume put -f RLHF ./deepspeed_config.json deepspeed_config.json 

modal volume put -f RLHF ./train_rm.sh train_rm.sh

echo "after"
modal volume ls RLHF

echo "done"
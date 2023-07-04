python ./run_ps.py \
    --name ps \
    --gpu_ids 4 5 6 7\
    2>&1 | tee logs/train_05.log

python ./run_ps2.py \
    --name ps2 \
    --gpu_ids 0 1 2 3\
    2>&1 | tee logs/train_ps2_01.log

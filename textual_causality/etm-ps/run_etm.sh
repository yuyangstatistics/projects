python ./run_etm.py \
    --name etm \
    --gpu_ids 4 5 6 7\
    2>&1 | tee logs/train_01.log

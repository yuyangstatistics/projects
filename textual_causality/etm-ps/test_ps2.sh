python ./test_ps2.py \
    --name ps2 \
    --gpu_ids 4 5 6 7\
    --split "train" \
    --ps_load_path "/home/yang6367/text-causal/etm/save/train/ps2-01/best.pth.tar" \
    2>&1 | tee logs/test_02.log

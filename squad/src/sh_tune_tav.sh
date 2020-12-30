python tune_tav.py \
    --split dev \
    --name tune \
    --gpu_ids 6 7 \
    --load_sketchy_path ./save/train/sreader-01/best.pth.tar \
    --load_intensive_path ./save/train/ireader-05/best.pth.tar 
    

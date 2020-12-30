python run_tav.py \
    --split dev \
    --threshold -0.006 \
    --name retro_reader \
    --gpu_ids 6 7 \
    --load_sketchy_path ./save/train/sreader-01/best.pth.tar \
    --load_intensive_path ./save/train/ireader-05/best.pth.tar 
    

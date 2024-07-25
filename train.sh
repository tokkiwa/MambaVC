CUDA_VISIBLE_DEVICES=0 python3 train.py --cuda -d /root/shared_smurai/datasets \
    -n 128 --lambda 0.05 --epochs 100 --lr_epoch 70 90 --batch-size 2 --save_path /root/shared_smurai/2024-MambaVC/
conda activate homefun
cd /Home/atr2/homefun/zhf/SS-Former
nohup python train.py --GPU 0 >weights/log_L_All.txt 2>&1 &
# source /Home/atr2/homefun/zhf/SS-Former/train_sh/train.sh
# CUDA_VISIBLE_DEVICES=3,6 nohup python -m torch.distributed.launch --nproc_per_node=2 train_mutil_GPU.py >weights/log_L_All.txt 2>&1 &
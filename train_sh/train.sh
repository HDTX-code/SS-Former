conda activate homefun
nohup python /Home/atr2/homefun/zhf/SS-Former/train.py --GPU 0 >weights/log_L_All.txt 2>&1 &
# source /Home/atr2/homefun/zhf/SS-Former/train_sh/train.sh
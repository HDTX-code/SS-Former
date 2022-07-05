# source /Home/atr2/homefun/zhf/SS-Former/train_sh/train_mutil_GPU.sh
conda activate homefun
# cd /devdata/home/homefun/SS-Former/
cd /Home/atr2/homefun/zhf/SS-Former
CUDA_VISIBLE_DEVICES=1,2 \
python -m torch.distributed.launch --nproc_per_node=2 \
train_mutil_GPU.py \
--Freeze_batch_size 148 \
--UnFreeze_batch_size 28 \
--train weights/2.5D/no_all/train.txt \
--val weights/2.5D/no_all/val.txt \
--cls_weights 0.3 0.7 | tee weights/logL4.txt 2>&1 &
cd src
# train
python main.py ctdet --exp_id coco_h85 --arch hardnet_85 --batch_size 48 --master_batch 24 --lr 1e-2 --gpus 0,1 --num_workers 16 --num_epochs 220 --lr_step 160,200
# The provided model was trained with Pytorch 1.5.0
# and two V100-32GB GPUs were used, where about 24GB momory will be consumed on each GPU
# test
python test.py ctdet --exp_id coco_h85 --arch hardnet_85 --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_h85 --arch hardnet_85 --keep_res --resume --flip_test

cd ..

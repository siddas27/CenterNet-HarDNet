cd src
# train
python main.py ctdet --exp_id coco_h85 --arch hardnet_85 --batch_size 48 --master_batch 11 --lr 1e-2 --gpus 0,1,2,3 --num_workers 16 --num_epochs 200 --lr_step 150,180
# test
python test.py ctdet --exp_id coco_h85 --arch hardnet_85 --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_h85 --arch hardnet_85 --keep_res --resume --flip_test

cd ..

#RGB
python main.py UESTC-MMEA-CL RGB --config ./exps/artf.json \
--train_list mydataset_train.txt --val_list mydataset_test.txt \
--arch ViT --num_segments 8 --dropout 0.5 --epochs 50 -b 16 \
--mpu_path '/data1/whx/temporal-binding-network/dataset/gyro/' \
--lr 0.001 --lr_steps 10 20 --gd 20 -j 8 --device '0' --freeze

# RGB & Gyrospec & Accespec
python main.py UESTC-MMEA-CL RGB Accespec Gyrospec --config ./exps/artf.json \
--train_list mydataset_train.txt --val_list mydataset_test.txt \
--arch ViT --num_segments 8 --dropout 0.5 --epochs 50 -b 16 \
--mpu_path '/data1/whx/temporal-binding-network/dataset/gyro/' \
--lr 0.001 --lr_steps 10 20 --gd 20 -j 8 --device '0' --freeze

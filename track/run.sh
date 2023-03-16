# TOGN
# train
python train_net_togn.py --pretrained ../data/basemodel_bit.pth --world_size 3 --device_list 0,1,2 --train_dropout_prob 0.5

# test
python train_net_togn.py --pretrained $your_model_path --world_size 1 --test --device_list 0 --master_port 12356 --batch_size 3


# CAGNet
# train
python train_net_cagnet.py --pretrained $your_model_path --world_size 3 --device_list 0,1,2 --master_port 12356 --batch_size 8

# test
python train_net_cagnet.py --test --pretrained $your_model_path --world_size 1 --device_list 2 --master_port 12356 --batch_size 3

# generate data
python generate_data.py

# train
python train_net_togn.py
python train_net_cagnet.py

# test
python train_net_togn.py --pretrained $your_model_path --test --world_size 3 
python train_net_cagnet.py --pretrained $your_model_path --test --world_size 3
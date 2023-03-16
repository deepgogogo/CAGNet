##  Consistency-Aware Graph Network for Human Interaction Understanding

### Update (2023.3.16)
- Add experiments on synthetic dataset which has a greater set of action classes (100 action classes).See `synthetic` directory for details.
- Construct Temporal Factor Graph Neural Network(TFGN) to leverage the temporal information in the videos. See `track` directory for details. 

### Introduction
Compared with the progress made on human activity
classification, much less success has been achieved on human interaction understanding (HIU). Apart from the latter
task is much more challenging, the main cause is that recent approaches learn human interactive relations via shallow graphical models, which is inadequate to model complicated human interactions. In this paper, we propose a
consistency-aware graph network, which combines the representative ability of graph network and the consistencyaware reasoning to facilitate HIU. Our network consists of
three components, a backbone CNN to extract image features, a factor graph network to learn third-order interactive relations among participants, and a consistency-aware
reasoning module to enforce labeling and grouping consistencies. Our key observation is that the consistency-awarereasoning bias for HIU can be embedded into an energy,
minimizing which delivers consistent predictions. An efficient mean-field inference algorithm is proposed, such that
all modules of our network could be trained jointly in an
end-to-end manner.

### Requirements
python >= 3.6
```commandline
pip install -r requirements.txt
```

### Download
Here we provide two datasets including BIT and TVHI (their copyrights belong to the original authors). Along with the datasets, we also provide
the pretrained basemodel weights and the final CAGNet model weights respectly. You can download them from

- BIT dataset [BaiduYun](https://pan.baidu.com/s/1hYQch02aJQN1dmWmQy25Yg) password: 4huw
- TVHI dataset [BaiduYun](https://pan.baidu.com/s/1f41VhH1LUlxrf1UhFqRnmw)  password: 0oia
- CAGNet_bit [BaiduYun](https://pan.baidu.com/s/18YAWt0Jgd9mhpAPOi9iOFg)  password: 06j0
- CAGNet_tvhi [BaiduYun](https://pan.baidu.com/s/12j9eZ4Wniit9vKOKbunqPA)  password: 3ii2
- **Basemodel_bit** [BaiduYun](https://pan.baidu.com/s/14KdWuVsrdZaR8rODDIKgiQ?pwd=6j9g) password: 6j9g 

After downloading these assets, put the model weights in `CAGNet/data` and extract the datasets to `CAGNet/data`. 
The default filenames should work properly. The directory `CAGNet/data` looks like this
```commandline
.
├── BIT
│   ├── BIT-anno
│   └── Bit-frames
├── bit.py
├── build_dataset.py
├── CAGNet_bit.pth
├── CAGNet_tvhi.pth
├── highfive
│   ├── frm
│   ├── readme.txt
│   └── tv_human_interaction_annotations
└── tvhi.py

```

### Training
We offer the training code on `BIT` dataset. To train the model, you should download the pretrained basemodel and put it 
into `data/` fold. Then run the program as follow:
```bash
cd cmd/
./train_bit.sh
```
Note that the default code is run on three GPUs, and you can adjust it in the scripts.

### Evaluation
Here we provide evaluation results same as in the paper.

The Evaluation bash scripts are in `cmd/`.

You can validate the CAGNet model of BIT by 
```commandline
cd cmd/ 
./eval_bit
```

You can validate the CAGNet model of TVHI by
```commandline
cd cmd/
./eval_tvhi
```

### Acknowledgement
We implement the factor graph based on [FGNN](https://github.com/zzhang1987/Factor-Graph-Neural-Network). We would like to express our sincere thanks to the contributors.

### Citation
If you find the code useful, please consider citing
```
@InProceedings{Wang_2021_ICCV,
author = {Wang, Zhenhua and Meng, Jiajun and Guo, Dongyan and Zhang, Jianhua and Javen Shi and Chen, Shengyong},
title = {Consistency-Aware Graph Network for Human Interaction Understanding},
booktitle = {ICCV},
month = {Oct},
year = {2021}
}
```

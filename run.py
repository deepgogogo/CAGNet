import sys
from train_net import train_net
from utils import setSeed
import datetime
import argparse
import os

setSeed(10)

# working_dir='/home/whz/CAGNet'
working_dir=os.path.split(os.path.abspath(__file__))[0]

parser=argparse.ArgumentParser()

# meanfield configs
parser.add_argument('--lambda_h',
                    type=float,
                    default=.5)

parser.add_argument('--lambda_g',
                    type=float,
                    default=.1)

parser.add_argument('--num_msg_time',
                    type=int,
                    default=6,
                    help='message times of meanfield')

# Training configs
parser.add_argument('--test',
                    action='store_true',
                    default=False,
                    help='set to evaluation mode')

parser.add_argument('--savedmodel_path',
                    type=str,
                    help='the model weight used to test')

parser.add_argument('--device_list',
                    type=str,
                    default='0,1,2',
                    help='set the used devices')

parser.add_argument('--use_multi_gpu',
                    type=bool,
                    default=True,
                    help='multi GPU mode')

parser.add_argument('--use_gpu',
                    type=bool,
                    default=True,
                    help='GPU mode')

parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    help='training batch size')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=32,
                    help='testing batch size')

parser.add_argument('--solver',
                    type=str,
                    default='adam',
                    help='optimizer')

parser.add_argument('--max_epoch',
                    type=int,
                    default=100,
                    help='max num of training epoch')

parser.add_argument('--test_interval_epoch',
                    type=int,
                    default=1,
                    help='frequency of test')

parser.add_argument('--base_model_path',
                    type=str,
                    default=f'',
                    help='path of pretrained base model path')

parser.add_argument('--log_path',
                    type=str,
                    default=f'{working_dir}/log',
                    help='path of log')

parser.add_argument('--save_model_path',
                    type=str,
                    default=f'{working_dir}/log',
                    help='path the model saved to')

parser.add_argument('--backbone',
                    type=str,
                    default='inception',
                    help='backbone')

parser.add_argument('--emb_features',
                    type=int,
                    default=1056,
                    help='feature dimention of backbone output')

parser.add_argument('--num_features_boxes',
                    type=int,
                    default=1024,
                    help='mid layer feature dimention, see it in gnn_model.py')

parser.add_argument('--linename',
                    type=str,
                    default='CAGNet',
                    help='name of the experiments, used by log and model name')


parser.add_argument('--train_learning_rate',
                    type=float,
                    default=0.0001,
                    help='learning rate')

parser.add_argument('--train_dropout_prob',
                    type=float,
                    default=0.1,
                    help='dropout probability')

parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0,
                    help='weight decay used for adam')

# Dataset configs

parser.add_argument('--dataset_name',
                    type=str,
                    default='',
                    choices=['bit','tvhi'],
                    help='choose the dataset for the experiment')

parser.add_argument('--num_boxes',
                    type=int,
                    default=15,
                    help='max num of person in one picture')

parser.add_argument('--num_actions',
                    type=int,
                    default=9,
                    help='max num of action in the dataset')

parser.add_argument('--action_weight',
                    nargs='+',
                    type=float,
                    default=None,
                    help='weight for the balance among classes')

parser.add_argument('--inter_weight',
                    nargs='+',
                    type=float,
                    default=None,
                    help='weight for the balance among classes')

parser.add_argument('--num_frames',
                    type=int,
                    default=1,
                    help='deprecated! just set to 1')

parser.add_argument('--num_workers',
                    type=int,
                    default=8,
                    help='num of workers of dataloader')

parser.add_argument('--image_size',
                    nargs='+',
                    type=int,
                    default=[540,960],
                    help='image size,H,w')

parser.add_argument('--out_size',
                    nargs='+',
                    type=int,
                    default=[65,117],
                    help='output size after passing through backbone')

parser.add_argument('--crop_size',
                    nargs='+',
                    type=int,
                    default=[5,5],
                    help='ROI crop size . H,W')


cfg=parser.parse_args()

print('Configs'.center(50,'='))

for k in list(vars(cfg).keys()):
    print('%s: %s' % (k, vars(cfg)[k]))

print('Configs'.center(50,'='))

train_net(cfg)


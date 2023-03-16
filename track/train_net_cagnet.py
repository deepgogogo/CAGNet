import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
import argparse

import time
import random
import os
from sklearn.metrics import  precision_recall_fscore_support,confusion_matrix,accuracy_score
import datetime
import itertools

from utils import AverageMeter,setSeed
from itertools import combinations,permutations
from CAGNet import BasenetFgnnMeanfield
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from bit import build_bit


def is_master():
    return dist.get_world_size()==1 or dist.get_rank()==0

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def createLogpath(cfg):
    if not is_master():return Path('')
    # eg CAGNet/log/bit/line_name
    log_path=os.path.join(
        cfg.log_path,
        f"{cfg.linename}_{datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}")

    if os.path.exists(log_path)==False:
        os.makedirs(log_path,exist_ok=True)

    # write configs
    with open(os.path.join(log_path,'config.cfg'),'a') as f:
        print('Configs'.center(50, '='),file=f)
        for k in list(vars(cfg).keys()):
            print(f"{k}: {vars(cfg)[k]}",file=f)
        print('Configs'.center(50, '='),file=f)
    print(log_path)
    return Path(log_path)

def write_log(log_path,msg_obj):
    if is_master():
        with open(log_path,'a') as f:
            print(json.dumps(msg_obj),file=f,end=',')
            
def gather_into_list(data_obj):
    """data_object is expected to be list of objs, returned value is 
    list of all objs across all gpus. numpy array as obj is recommended"""
    world_size=dist.get_world_size()
    if world_size==1:
        return data_obj
    outputs=[None for _ in range(world_size)]
    dist.all_gather_object(outputs, data_obj)
    data_obj=list(itertools.chain.from_iterable(outputs))
    return data_obj
            

def train_net(cfg):
    rank=dist.get_rank()
    
    log_path=createLogpath(cfg)

    # Reading dataset
    training_loader,validation_loader=build_bit(cfg)

    model=BasenetFgnnMeanfield(cfg)
    
    if cfg.test==True:
        model.load_state_dict(torch.load(cfg.pretrained))
        model = model.to(rank)
        model=DDP(model,device_ids=[rank],output_device=rank)
        
        if is_master():print('begin test')
        test_info = test_func(validation_loader, model, 'test', cfg)
        write_log(log_path/'test_log.json',test_info)
        if is_master():print('end test')
        exit(0)
        
    if cfg.pretrained!=None:
        model.load_state_dict(torch.load(cfg.pretrained),strict=False)
    
    model = model.to(rank)
    model=DDP(model,device_ids=[rank],output_device=rank,find_unused_parameters=True)
    model_noddp=model.module
    

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),
                           lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)
    
    # for debug
    # test_info = test_func(validation_loader, model, 'test', cfg)
    # write_log(log_path/'test_log',test_info)
    # exit(0)
    
    # train_info=train_func(training_loader, model, optimizer, 'train', cfg)
    # write_log(log_path/'train_log',train_info)
    # if is_master():
    #     torch.save(model_noddp.state_dict(),log_path/'train_debug.pth')
    # exit(0)
    
    for epoch in range(cfg.max_epoch):
        training_loader.sampler.set_epoch(epoch)
        train_info=train_func(training_loader, model, optimizer, epoch, cfg)
        
        write_log(log_path/'train_log.json',train_info)
                
        if epoch % cfg.test_interval_epoch == 0:# evaluation
            test_info = test_func(validation_loader, model, epoch, cfg)
            write_log(log_path/'test_log.json',test_info)
            if is_master():
                torch.save(model_noddp.state_dict(),log_path/f'epoch{epoch}.pth')
                print('*epoch ' + str(epoch) + ' finished')
    pass


def train_func(data_loader, model, optimizer, epoch, cfg):
    loss_meter = AverageMeter()
    rank=dist.get_rank()
    
    losses,act_losses,inter_losses=[],[],[]
    model.train()
    for samples in data_loader:
        
        act_score,inter_score=model(samples)
        
        batch_sz,clip_sz=samples['frame_volumn'].shape[:2]
        action=samples['action_volumn'].to(rank)
        interaction=samples['interaction_volumn'].to(rank)
        box_num=samples['box_num_volumn'].to(rank)
        key_frame=samples['key_frame'].squeeze(-1)
        
        actions,interactions=[],[]
        for b,t in enumerate(key_frame):
            n=box_num[b,t]
            actions.append(action[b,t,:n])
            interactions.append(interaction[b,t,:n*(n-1)])

        actions=torch.cat(actions)
        interactions=torch.cat(interactions)
        
        act_loss = F.cross_entropy(act_score,actions)  # cross_entropy
        inter_loss = F.cross_entropy(inter_score,interactions)
        total_loss = act_loss + inter_loss

        loss_meter.update(val=total_loss.item(), n=1)
        
        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        act_losses.append(act_loss.item())
        inter_losses.append(inter_loss.item())
        
    losses=np.array(gather_into_list(losses)).mean()
    act_losses=np.array(gather_into_list(act_losses)).mean()
    inter_losses=np.array(gather_into_list(inter_losses)).mean()
    return {'epoch':epoch,'loss':losses,'act_loss':act_losses,'inter_loss':inter_losses}
    

@torch.no_grad()
def test_func(data_loader, model, epoch, cfg):
    loss_meter = AverageMeter()
    rank=dist.get_rank()
    world_size=dist.get_world_size()
    model.eval()
    
    act_groundtruth=[]
    act_predict=[]
    inter_groundtruth=[]
    inter_predict=[]
    
    for samples in data_loader:
        
        act_score,inter_score=model(samples)
        
        batch_sz,clip_sz=samples['frame_volumn'].shape[:2]
        action=samples['action_volumn'].to(rank)
        interaction=samples['interaction_volumn'].to(rank)
        box_num=samples['box_num_volumn'].to(rank)
        key_frame=samples['key_frame'].squeeze(-1)
        
        actions,interactions=[],[]
        ahead,ihead=0,0
        for b,t in enumerate(key_frame):
            n=box_num[b,t]
            act_groundtruth.append(action[b,t,:n].cpu().numpy())
            inter_groundtruth.append(interaction[b,t,:n*(n-1)].cpu().numpy())
            act_predict.append(torch.argmax(torch.softmax(act_score[ahead:ahead+n],dim=-1),dim=-1).cpu().numpy())
            inter_predict.append(torch.argmax(torch.softmax(inter_score[ihead:ihead+n*(n-1)],dim=-1),dim=-1).cpu().numpy())
            ahead+=n
            ihead+=n*(n-1)

        # actions=torch.cat(actions)
        # interactions=torch.cat(interactions)
        
        # act_groundtruth.append(actions.cpu().numpy())
        # act_predict.append(torch.argmax(torch.softmax(act_score,dim=-1),dim=-1).cpu().numpy())
        # inter_groundtruth.append(interactions.cpu().numpy())
        # inter_predict.append(torch.argmax(torch.softmax(inter_score,dim=-1),dim=-1).cpu().numpy())
        
        # for test
        # act_score=torch.zeros((len(actions),9),device=rank)
        # act_score[torch.arange(len(actions),device=rank) ,actions]=1
        # inter_score=torch.zeros((len(interactions),2),device=rank)
        # inter_score[torch.arange(len(interactions),device=rank),interactions]=1
        
        # act_loss = F.cross_entropy(act_score,actions)  # cross_entropy
        # inter_loss = F.cross_entropy(inter_score,interactions)
        # total_loss = act_loss + inter_loss

        # loss_meter.update(val=total_loss.item(), n=1)
        
        
    act_groundtruth=gather_into_list(act_groundtruth)
        
    act_predict=gather_into_list(act_predict)
        
    inter_groundtruth=gather_into_list(inter_groundtruth)

    inter_predict=gather_into_list(inter_predict)
    
    metric={}
    if is_master():
        # print(loss_meter.avg)
        uncompat,untrans=count_conflict(act_predict,inter_predict,data_loader.dataset.conflict_matrix)
            
        act_predict=np.concatenate(act_predict)
        act_groundtruth=np.concatenate(act_groundtruth)
        act_accuracy=accuracy_score(act_groundtruth,act_predict)
        actions_precision, actions_recall, actions_F1, support = precision_recall_fscore_support(act_groundtruth,act_predict,beta=1, average='macro')
    
        inter_predict=np.concatenate(inter_predict)
        inter_groundtruth=np.concatenate(inter_groundtruth)
        inter_accuracy=accuracy_score(inter_groundtruth,inter_predict)
        inter_precision, inter_recall, inter_F1, support = precision_recall_fscore_support(inter_groundtruth,inter_predict,beta=1, average='macro')
        
        conf_mat=confusion_matrix(act_groundtruth, act_predict)
        conf_mat = conf_mat/np.expand_dims(np.sum(conf_mat, axis=1), axis=1)
        
        # calculate mean IOU
        cls_iou = np.array([0 for _ in range(cfg.num_actions + 2)],dtype=float)
        for i in range(cfg.num_actions):
            grd = set((act_groundtruth == i).nonzero()[0].tolist())
            prd = set((act_predict == i).nonzero()[0].tolist())
            uset = grd.union(prd)
            iset = grd.intersection(prd)
            cls_iou[i] = len(iset) / len(uset)
        for i in range(2):
            grd = set((inter_groundtruth == i).nonzero()[0].tolist())
            prd = set((inter_predict == i).nonzero()[0].tolist())
            uset = grd.union(prd)
            iset = grd.intersection(prd)
            cls_iou[cfg.num_actions + i] = len(iset) / len(uset)
        mean_iou = cls_iou.mean()
        
        metric={
            'epoch':epoch,
            'act_accuracy':act_accuracy,
            'actions_precision':actions_precision,
            'actions_recall':actions_recall,
            'actions_F1':actions_F1,
            'inter_accuracy':inter_accuracy,
            'inter_precision':inter_precision,
            'inter_recall':inter_recall,
            'inter_F1':inter_F1,
            'uncompat':uncompat.item(),
            'untrans':untrans.item(),
            'both_acc':(act_accuracy+inter_accuracy)/2,
            'both_F1':(actions_F1+inter_F1)/2,
            'mean_IOU':mean_iou,
            'conf_mat':conf_mat.tolist(),
            'lambda_h':model.module.lambda_h.item(),
            'lambda_g':model.module.lambda_g.item(),
        }
    return metric


def count_conflict(act_list,inter_list,conflict_table):
    c_cnt,t_cnt=0,0
    for act,inter in zip(act_list,inter_list):
        r=count_oneframe_conflict(act,inter,conflict_table)
        c_cnt+=r[0]
        t_cnt+=r[1]
    return c_cnt,t_cnt 

def count_oneframe_conflict(act,inter,conflict_table):
    
    N=act.shape[0]
    rmat=np.zeros((N,N),dtype=inter.dtype)
    ulidx=np.array([p for p in permutations(range(N),2)])
    rmat[ulidx[:,0],ulidx[:,1]]=inter

    # get action label
    act_label=act[np.array(np.nonzero(rmat)).T]
    uncompat_cnt,untrans_cnt=0,0
    if act_label.shape[1]>0:
        uncompat_cnt=np.sum(conflict_table[act_label[:,0],act_label[:,1]])
    if N>2: 
        grp=[]
        for c in combinations(range(3),3):
            grp.append([p for p in combinations(c,2)])
        grp=np.array(grp)
        inter_grp=rmat[grp[:,:,0],grp[:,:,1]]
        inter_num=np.sum(inter_grp,axis=1)
        untrans_cnt=np.sum(inter_num==2)
        
    return uncompat_cnt,untrans_cnt


def cmdline():
    setSeed(10)
    
    working_dir=os.path.split(os.path.abspath(__file__))[0]

    parser=argparse.ArgumentParser()

    # dist
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--world_size",type=int, default=1
    )
    parser.add_argument(
        '--master_port',
        type=str,
        default='12355',
    )

    # model config

    parser.add_argument('--num_fgnn_layer',
                        type=int,
                        default=10,
                        help='')

    parser.add_argument('--lambda_h',
                        type=float,
                        default=.5)

    parser.add_argument('--lambda_g',
                        type=float,
                        default=.1)
    
    parser.add_argument('--num_msg_time',
                    type=int,
                    default=6)

    # Training configs
    parser.add_argument('--test',
                        action='store_true',
                        default=False,
                        help='set to evaluation mode')

    parser.add_argument('--pretrained',
                        type=str,
                        help='the model weight used to test')

    parser.add_argument('--device_list',
                        type=str,
                        default='0',
                        help='set the used devices')

    parser.add_argument('--use_multi_gpu',
                        action='store_true',
                        help='multi GPU mode')

    parser.add_argument('--use_gpu',
                        action='store_true',
                        help='GPU mode')

    parser.add_argument('--batch_size',
                        type=int,
                        default=3,
                        help='training batch size')

    parser.add_argument('--max_epoch',
                        type=int,
                        default=100,
                        help='max num of training epoch')

    parser.add_argument('--test_interval_epoch',
                        type=int,
                        default=1,
                        help='frequency of test')

    parser.add_argument('--log_path',
                        type=str,
                        default=f'{working_dir}/log',
                        help='path of log')

    parser.add_argument('--save_model_path',
                        type=str,
                        default=f'{working_dir}/log',
                        help='path the model saved to')

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
    
    parser.add_argument('--emb_features',
                        type=int,
                        default=1056,
                        help='feature dimention of backbone output')

    parser.add_argument('--num_features_boxes',
                        type=int,
                        default=1024,
                        help='mid layer feature dimention, see it in gnn_model.py')

    # Dataset configs
    
    parser.add_argument('--dataset_name',
                        type=str,
                        default='bit')

    parser.add_argument('--num_boxes',
                        type=int,
                        default=5,
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

    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
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
    return cfg

def dist_setup(rank, world_size,addr='localhost',port='12355'):
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def main(rank,args):
    torch.cuda.set_device(rank)
    world_size=args.world_size
    dist_setup(rank,world_size,port=args.master_port)
    train_net(args)
    dist.destroy_process_group()
    pass

if __name__=='__main__':
    args=cmdline()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_list
    # num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    # assert num_gpu <= torch.cuda.device_count()
    mp.spawn(main,args=[args],nprocs=args.world_size)
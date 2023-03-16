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

from synthetic_dataset import build_synthetics
from utils import AverageMeter,setSeed
from itertools import combinations,permutations
from togn_model import BasenetFgnn
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


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
            

def train_net(cfg):
    rank=dist.get_rank()
    
    log_path=createLogpath(cfg)

    # Reading dataset
    training_loader,validation_loader=build_synthetics()

    model=BasenetFgnn(cfg)
    
    if cfg.test==True:
        model.load_state_dict(torch.load(cfg.pretrained))
        model = model.to(rank)
        model=DDP(model,device_ids=[rank],output_device=rank)
        
        if is_master():print('begin test')
        test_info = test_func(validation_loader, model, 'test', cfg)
        write_log(log_path/'test_log.json',test_info)
        if is_master():print('end test')
        exit(0)
    
    model = model.to(rank)
    model=DDP(model,device_ids=[rank],output_device=rank)
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
        
        write_log(log_path/'train_log',train_info)
                
        if epoch % cfg.test_interval_epoch == 0:# evaluation
            test_info = test_func(validation_loader, model, epoch, cfg)
            write_log(log_path/'test_log',test_info)
            if is_master():
                torch.save(model_noddp.state_dict(),log_path/f'epoch{epoch}.pth')
                print('*epoch ' + str(epoch) + ' finished')
    pass


def train_func(data_loader, model, optimizer, epoch, cfg):
    loss_meter = AverageMeter()
    rank=dist.get_rank()
    world_size=dist.get_world_size()
    
    act_groundtruth=[]
    act_predict=[]
    inter_groundtruth=[]
    inter_predict=[]
    
    model.train()
    for score, target in data_loader:
        n_person=target['n_person']
        inter_score=score['pair_inter_score']
        inter_score=inter_score[:,~np.eye(n_person,n_person,dtype=bool),:].float().squeeze().to(rank)
        act_score=score['act_score'].float().squeeze().to(rank)
        
        act_label=target['act_label'].squeeze().to(rank)
        inter_label=target['pair_inter_label']
        inter_label=inter_label[:,~np.eye(n_person,n_person,dtype=bool)].squeeze().to(rank)
        
        act_score,inter_score=model((act_score,inter_score,n_person))
        
        act_loss = F.cross_entropy(act_score,act_label)  # cross_entropy
        inter_loss = F.cross_entropy(inter_score,inter_label)
        total_loss = act_loss + inter_loss

        loss_meter.update(val=total_loss.item(), n=1)
        
        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        act_groundtruth.append(act_label.cpu().numpy())
        # act_predict.append(torch.argmax(act_score,dim=-1).cpu().numpy())
        act_predict.append(torch.argmax(torch.softmax(act_score,dim=-1),dim=-1).cpu().numpy())
        inter_groundtruth.append(inter_label.cpu().numpy())
        # inter_predict.append(torch.argmax(inter_score,dim=-1).cpu().numpy())
        inter_predict.append(torch.argmax(torch.softmax(inter_score,dim=-1),dim=-1).cpu().numpy())
        
    outputs=[None for _ in range(world_size)]
    dist.all_gather_object(outputs, act_groundtruth)
    if is_master():
        act_groundtruth=list(itertools.chain.from_iterable(outputs))
        
    outputs=[None for _ in range(world_size)]
    dist.all_gather_object(outputs, act_predict)
    if is_master():
        act_predict=list(itertools.chain.from_iterable(outputs))
        
    outputs=[None for _ in range(world_size)]
    dist.all_gather_object(outputs, inter_groundtruth)
    if is_master():
        inter_groundtruth=list(itertools.chain.from_iterable(outputs))

    outputs=[None for _ in range(world_size)]
    dist.all_gather_object(outputs, inter_predict)
    if is_master():
        inter_predict=list(itertools.chain.from_iterable(outputs))
        
    metric={}
    if is_master():
        # print(loss_meter.avg)
        uncompat,untrans=count_conflict(act_predict,inter_predict,data_loader.dataset.conflict_matrix)
        print(f'uncompat={uncompat},untrans={untrans}')
            
        act_predict=np.concatenate(act_predict)
        act_groundtruth=np.concatenate(act_groundtruth)
        act_accuracy=accuracy_score(act_groundtruth,act_predict)
        actions_precision, actions_recall, actions_F1, support = precision_recall_fscore_support(act_groundtruth,act_predict,beta=1, average='macro')
        print(f'accuray={act_accuracy}\nprecision={actions_precision}\nrecall={actions_recall}\nF1={actions_F1}\n')
        
        inter_predict=np.concatenate(inter_predict)
        inter_groundtruth=np.concatenate(inter_groundtruth)
        inter_accuracy=accuracy_score(inter_groundtruth,inter_predict)
        inter_precision, inter_recall, inter_F1, support = precision_recall_fscore_support(inter_groundtruth,inter_predict,beta=1, average='macro')
        print(f'accuray={inter_accuracy}\nprecision={inter_precision}\nrecall={inter_recall}\nF1={inter_F1}\n')
    
        print('epoch: ' + str(epoch) + ', loss: ' + str(loss_meter.avg))
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
            
        }
    return metric
    

@torch.no_grad()
def test_func(data_loader, model, epoch, cfg):
    loss_meter = AverageMeter()
    rank=dist.get_rank()
    world_size=dist.get_world_size()
    num_actions=cfg.num_actions
    model.eval()
    
    act_groundtruth=[]
    act_predict=[]
    inter_groundtruth=[]
    inter_predict=[]
    
    for i,(score, target) in enumerate(data_loader):
        n_person=target['n_person']
        inter_score=score['pair_inter_score']
        inter_score=inter_score[:,~np.eye(n_person,n_person,dtype=bool),:].float().squeeze().to(rank)
        act_score=score['act_score'].float().squeeze().to(rank)
        
        act_label=target['act_label'].squeeze().to(rank)
        inter_label=target['pair_inter_label']
        inter_label=inter_label[:,~np.eye(n_person,n_person,dtype=bool)].squeeze().to(rank)
        
        act_score,inter_score=model((act_score,inter_score,n_person))
        
        act_loss = F.cross_entropy(act_score,act_label)  # cross_entropy
        inter_loss = F.cross_entropy(inter_score,inter_label)
        total_loss = act_loss + inter_loss

        loss_meter.update(val=total_loss.item(), n=1)
        
        act_groundtruth.append(act_label.cpu().numpy())
        # act_predict.append(torch.argmax(act_score,dim=-1).cpu().numpy())
        act_predict.append(torch.argmax(torch.softmax(act_score,dim=-1),dim=-1).cpu().numpy())
        inter_groundtruth.append(inter_label.cpu().numpy())
        # inter_predict.append(torch.argmax(inter_score,dim=-1).cpu().numpy())
        inter_predict.append(torch.argmax(torch.softmax(inter_score,dim=-1),dim=-1).cpu().numpy())
        
    outputs=[None for _ in range(world_size)]
    dist.all_gather_object(outputs, act_groundtruth)
    if is_master():
        act_groundtruth=list(itertools.chain.from_iterable(outputs))
        
    outputs=[None for _ in range(world_size)]
    dist.all_gather_object(outputs, act_predict)
    if is_master():
        act_predict=list(itertools.chain.from_iterable(outputs))
        
    outputs=[None for _ in range(world_size)]
    dist.all_gather_object(outputs, inter_groundtruth)
    if is_master():
        inter_groundtruth=list(itertools.chain.from_iterable(outputs))

    outputs=[None for _ in range(world_size)]
    dist.all_gather_object(outputs, inter_predict)
    if is_master():
        inter_predict=list(itertools.chain.from_iterable(outputs))
    
    metric={}
    if is_master():
        # print(loss_meter.avg)
        uncompat,untrans=count_conflict(act_predict,inter_predict,data_loader.dataset.conflict_matrix)
        print(f'uncompat={uncompat},untrans={untrans}')
            
        act_predict=np.concatenate(act_predict)
        act_groundtruth=np.concatenate(act_groundtruth)
        act_accuracy=accuracy_score(act_groundtruth,act_predict)
        actions_precision, actions_recall, actions_F1, support = precision_recall_fscore_support(act_groundtruth,act_predict,beta=1, average='macro')
        act_f1s=precision_recall_fscore_support(act_groundtruth,act_predict,beta=1, average=None)[2]
        print(f'accuray={act_accuracy}\nprecision={actions_precision}\nrecall={actions_recall}\nF1={actions_F1}\n')
    
        inter_predict=np.concatenate(inter_predict)
        inter_groundtruth=np.concatenate(inter_groundtruth)
        inter_accuracy=accuracy_score(inter_groundtruth,inter_predict)
        inter_precision, inter_recall, inter_F1, support = precision_recall_fscore_support(inter_groundtruth,inter_predict,beta=1, average='macro')
        print(f'accuray={inter_accuracy}\nprecision={inter_precision}\nrecall={inter_recall}\nF1={inter_F1}\n')
        
        # calculate mean IOU
        cls_iou = np.array([0 for _ in range(num_actions + 2)],dtype=float)
        for i in range(num_actions):
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
            cls_iou[num_actions + i] = len(iset) / len(uset)
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
            'act_f1s':act_f1s.tolist(),
            'F1':(actions_F1+inter_F1)/2,
            'accuracy':(act_accuracy+inter_accuracy)/2,
            'mean_iou':mean_iou,
            'inconsistency':(uncompat.item()+untrans.item()),
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
        "--world_size",type=int, default=3
    )

    # model config

    parser.add_argument('--num_fgnn_layer',
                        type=int,
                        default=10,
                        help='')

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
                        default=1,
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
                        default='TOGN',
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

    parser.add_argument('--num_boxes',
                        type=int,
                        default=25,
                        help='max num of person in one picture')

    parser.add_argument('--num_actions',
                        type=int,
                        default=100,
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

    cfg=parser.parse_args()

    print('Configs'.center(50,'='))

    for k in list(vars(cfg).keys()):
        print('%s: %s' % (k, vars(cfg)[k]))

    print('Configs'.center(50,'='))
    return cfg

def dist_setup(rank, world_size,addr='localhost',port='12356'):
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def main(rank,args):
    torch.cuda.set_device(rank)
    world_size=args.world_size
    dist_setup(rank,world_size)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_list
    train_net(args)
    dist.destroy_process_group()
    pass

if __name__=='__main__':
    args=cmdline()
    # num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    # assert num_gpu <= torch.cuda.device_count()
    mp.spawn(main,args=[args],nprocs=args.world_size)
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import  precision_recall_fscore_support,confusion_matrix
import datetime

from CAGNet import BasenetFgnnMeanfield
from data.build_dataset import build_dataset
from utils import *


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def createLogpath(cfg):
    # eg CAGNet/log/bit/line_name
    log_path=os.path.join(
        cfg.log_path,
        cfg.dataset_name,
        f"{cfg.linename}_{datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}")

    if os.path.exists(log_path)==False:
        os.makedirs(log_path,exist_ok=True)

    return log_path

def test_net(cfg):
    """
    training net
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    log_path=createLogpath(cfg)
    print(log_path)

    # Reading dataset
    training_loader,validation_loader=build_dataset(cfg)


    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model=BasenetFgnnMeanfield(cfg)

    if cfg.use_multi_gpu:
        model = nn.DataParallel(model)

    model = model.cuda()
    model.train()  # train mode
    model.apply(set_bn_eval)

    if cfg.solver == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=cfg.train_learning_rate,
                               weight_decay=cfg.weight_decay)


    if cfg.test==True:
        print('begin test')
        test_info = test_func(validation_loader, model, device, optimizer, 'test', cfg)
        save_log(cfg,log_path,'test_info', test_info)
        print('end test')
        exit(0)

    raise NotImplementedError


def test_func(data_loader, model, device, optimizer, epoch, cfg):
    model.eval()

    actions_meter = AverageMeter()
    interactions_meter = AverageMeter()
    loss_meter = AverageMeter()
    actions_classification_labels = [0 for i in range(cfg.num_actions)]
    actions_classification_pred_true = [0 for i in range(cfg.num_actions)]
    interactions_classification_labels = [0, 0]
    interactions_classification_pred_true = [0, 0]

    actions_pred_global = []
    actions_labels_global = []
    interactions_pred_global = []
    interactions_labels_global = []

    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            seq_name, fid = batch_data[-2], batch_data[-1]
            batch_data = [b.to(device=device) for b in batch_data[0:-2]]
            batch_size = batch_data[0].shape[0]
            num_frames = batch_data[0].shape[1]

            actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))
            interactions_in = batch_data[4].reshape((batch_size, num_frames, cfg.num_boxes * (cfg.num_boxes - 1)))

            bboxes_num = batch_data[3].reshape(batch_size, num_frames)

            # forward
            actions_scores, interactions_scores = model((batch_data[0], batch_data[1], batch_data[3]))

            actions_in_nopad = []
            interactions_in_nopad = []

            actions_in = actions_in.reshape((batch_size * num_frames, cfg.num_boxes,))
            interactions_in = interactions_in.reshape(
                (batch_size * num_frames, cfg.num_boxes * (cfg.num_boxes - 1),))
            bboxes_num = bboxes_num.reshape(batch_size * num_frames, )
            for bt in range(batch_size * num_frames):
                N = bboxes_num[bt]
                actions_in_nopad.append(actions_in[bt, :N])
                interactions_in_nopad.append(interactions_in[bt, :N * (N - 1)])

            actions_in = torch.cat(actions_in_nopad, dim=0).reshape(-1, )  # ALL_N,
            interactions_in = torch.cat(interactions_in_nopad, dim=0).reshape(-1, )

            aweight, iweight = None, None
            if cfg.action_weight != None:
                aweight = torch.tensor(cfg.action_weight,
                                       dtype=torch.float,
                                       device='cuda')
            if cfg.inter_weight != None:
                iweight = torch.tensor(cfg.inter_weight,
                                       dtype=torch.float,
                                       device='cuda')

            actions_loss = F.cross_entropy(actions_scores,
                                           actions_in,
                                           weight=aweight)

            interactions_loss = F.cross_entropy(interactions_scores,
                                                interactions_in,
                                                weight=iweight)

            actions_pred = torch.argmax(actions_scores, dim=1)  # ALL_N,
            actions_correct = torch.sum(torch.eq(actions_pred.int(), actions_in.int()).float())

            interactions_pred = torch.argmax(interactions_scores, dim=1)
            interactions_correct = torch.sum(torch.eq(interactions_pred.int(), interactions_in.int()).float())

            actions_pred_global.append(actions_pred.cpu())
            actions_labels_global.append(actions_in.cpu())
            interactions_pred_global.append(interactions_pred.cpu())
            interactions_labels_global.append(interactions_in.cpu())

            # calculate recall
            for i in range(len(actions_pred)):
                actions_classification_labels[actions_in[i]] += 1
                if actions_pred[i] == actions_in[i]:
                    actions_classification_pred_true[actions_pred[i]] += 1
            for i in range(len(interactions_pred)):
                interactions_classification_labels[interactions_in[i]] += 1
                if interactions_pred[i] == interactions_in[i]:
                    interactions_classification_pred_true[interactions_pred[i]] += 1
            # Get accuracy
            actions_accuracy = \
                actions_correct.item() / actions_scores.shape[0]
            interactions_accuracy = \
                interactions_correct.item() / interactions_in.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            interactions_meter.update(interactions_accuracy, interactions_in.shape[0])

            # Total loss
            total_loss = actions_loss + interactions_loss
            loss_meter.update(total_loss.item(), batch_size)

    for i in range(len(actions_classification_labels)):
        actions_classification_pred_true[i] = \
            actions_classification_pred_true[i] * 1.0 / actions_classification_labels[i]
    for i in range(len(interactions_classification_labels)):
        interactions_classification_pred_true[i] = \
            interactions_classification_pred_true[i] * 1.0 / interactions_classification_labels[i]

    actions_pred_global = torch.cat(actions_pred_global)
    actions_labels_global = torch.cat(actions_labels_global)
    interactions_pred_global = torch.cat(interactions_pred_global)
    interactions_labels_global = torch.cat(interactions_labels_global)

    # calculate mean IOU
    cls_iou = torch.Tensor([0 for _ in range(cfg.num_actions + 2)]).cuda().float()
    for i in range(cfg.num_actions):
        grd = set((actions_labels_global == i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())
        prd = set((actions_pred_global == i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())
        uset = grd.union(prd)
        iset = grd.intersection(prd)
        cls_iou[i] = len(iset) / len(uset)
    for i in range(2):
        grd = set((interactions_labels_global == i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())
        prd = set((interactions_pred_global == i).nonzero().squeeze(dim=-1).cpu().numpy().tolist())
        uset = grd.union(prd)
        iset = grd.intersection(prd)
        cls_iou[cfg.num_actions + i] = len(iset) / len(uset)
    mean_iou = cls_iou.mean()

    actions_precision, actions_recall, actions_F1, support = precision_recall_fscore_support(actions_labels_global,
                                                                                             actions_pred_global,
                                                                                             beta=1, average='macro')
    interactions_precision, interactions_recall, interactions_F1, support = precision_recall_fscore_support(
        interactions_labels_global, interactions_pred_global, beta=1, average='macro')

    conf_mat=confusion_matrix(actions_labels_global.cpu().numpy(),actions_pred_global.cpu().numpy())
    conf_mat = conf_mat/np.expand_dims(np.sum(conf_mat, axis=1), axis=1)

    test_info = {
        # 'time':epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        # 'activities_acc':activities_meter.avg*100,
        'actions_precision': actions_precision,
        'actions_recall': actions_recall,
        'actions_F1': actions_F1,
        'actions_acc': actions_meter.avg * 100,
        'actions_classification_recalls': actions_classification_pred_true,
        'interactions_precision': interactions_precision,
        'interactions_recall': interactions_recall,
        'interactions_F1': interactions_F1,
        'interactions_acc': interactions_meter.avg * 100,
        'interactions_classification_recalls': interactions_classification_pred_true,
        'mean_iou': mean_iou,
        'confusion_matrix':conf_mat
    }

    return test_info

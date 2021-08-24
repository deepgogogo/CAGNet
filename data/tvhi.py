from torch.utils import data
import torchvision.transforms as transforms
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *

# import random
from PIL import Image
import numpy as np
import re
import os.path


action_dict={"hand_shake":0,"high_five":1,"hug":2,"kiss":3,"no_interaction":4}

anno_file_set1 = {"handShake":None, "highFive":None, "hug":None, "kiss":None, "negative":None}
anno_file_set2 = {"handShake":None, "highFive":None, "hug":None, "kiss":None, "negative":None}

# CAGNet/data/highfive
data_path=os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        'highfive')

for k, v in anno_file_set1.items():
    if k is not "negative":
        anno_file_set1.update({k:list(range(1,36))})
        anno_file_set2.update({k: list(range(36,51))})
    else: # negative
        anno_file_set1.update({k:list(range(1,36))})
        anno_file_set2.update({k: list(range(36,51))})

def get_train_test_seq():
    trn_seq,tst_seq=[],[]
    for k,v in anno_file_set1.items():
        for id in v:
            trn_seq.append('{}_{:04d}'.format(k,id))

    for k,v in anno_file_set2.items():
        for id in v:
            tst_seq.append('{}_{:04d}'.format(k,id))

    return trn_seq,tst_seq

def TVHI_read_annotations(sname):
    annotations={}
    ann_path=data_path+"/tv_human_interaction_annotations/"+sname+".annotations"
    img_path=data_path+"/frm"
    ptr=0
    with open(ann_path,mode='r') as f:
        fid,box_num,pic_path=0,0,""
        u,v=-1,-1
        id={}
        for line in f:
            t=re.split(' +',line.strip())
            if len(t)==2:continue# file head,`num_frames:46`,eg
            if t[0]=="#frame:":# frame head,`#frame:0  #num_bbxs:2`,eg
                fid=int(t[1])
                pic_path=img_path+"/"+sname+"/{:04d}.jpg".format(fid)
                box_num=int(t[3])
                if os.path.exists(pic_path) == False or box_num<=1: continue
                ptr=0
                if len(t)>4:
                    u,v=int(t[-3]),int(t[-1])
                annotations[fid]={
                    'frame_id':fid,
                    'actions':[],
                    'interactions':[],
                    'box_num':box_num,
                    'bboxes':[]
                }
            else:
                if os.path.exists(pic_path)==False or box_num<=1: continue
                annotations[fid]['actions'].append(action_dict[t[4]])
                id[int(t[0])]=ptr
                x1,y1=int(t[1]),int(t[2])
                x2,y2=x1+int(t[3]),y1+int(t[3])
                annotations[fid]['bboxes'].append([y1,x1,y2,x2])
                ptr+=1
            if ptr==box_num:
                intrct = []
                if u!=-1 and v!=-1:
                    u=id[u]
                    v=id[v]
                    for i in range(box_num):
                        for j in range(box_num):
                            if i==j:continue
                            if (i==u and j==v) or (i==v and j==u):
                                intrct.append(1)
                            else:intrct.append(0)
                if len(intrct)==0:
                    intrct=[0 for i in range(box_num*(box_num-1))]
                annotations[fid]['interactions']=intrct
                ptr=0
                fid, box_num, pic_path = 0, 0, ""
                u, v = -1, -1
                id = {}
        #
        # remove frames without interaction
        for k, v in list(annotations.items()):
            if sum(v['interactions']) == 0:
                annotations.pop(k)
    return annotations

def TVHI_read_dataset(seqs):
    data={}
    for sname in seqs:
        data[sname]=TVHI_read_annotations(sname)
    return data

def TVHI_all_frames(anns):
    return [(s,f) for s in anns for f in anns[s]]#s:seq_name,f:frame_id

class TVHIDataset(data.Dataset):
    def __init__(self, anns,
                 frames,
                 image_size,
                 feature_size,
                 num_boxes=5,
                 num_frames=1,
                 data_augmentation=False):
        '''anns: return by BIT_read_dataset
           frames: return by BIT_all_frames
           feature_size: the output feature map size of backbone
        '''
        self.anns = anns
        self.frames = frames
        self.image_path = data_path + '/frm'
        self.image_size = image_size
        self.feature_size = feature_size

        self.num_boxes = num_boxes
        self.num_frames = num_frames

        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        sample=self.load_samples_sequence(self.frames[index])
        return  sample

    def load_samples_sequence(self,select_frame):
        random_factor = np.random.randint(0, 2) #随机数0,1
        OH, OW = self.feature_size
        images, bboxes = [], []
        actions = []
        interactions = []
        bboxes_num = []

        seq_name, fid = select_frame
        name_id = seq_name.split('_')
        img = Image.open(self.image_path + '/' + seq_name + '/' + '{:04d}.jpg'.format(fid))
        W,H=img.size[0],img.size[1]
        temp_boxes = []
        anno_box=self.anns[seq_name][fid]['bboxes']
        anno_box=np.array(anno_box)
        sz = anno_box[:, 2] - anno_box[:, 0]
        sz=sz/4.0
        anno_box[:,0]=anno_box[:,0]-1*sz
        anno_box[:,0][anno_box[:,0]<0]=0
        anno_box[:,1]=anno_box[:,1]-1.5*sz
        anno_box[:,1][anno_box[:,1]<0]=0
        anno_box[:,2]=anno_box[:,2]+2.0*sz
        anno_box[:,2][anno_box[:,2]>H]=H
        anno_box[:,3]=anno_box[:,3]+1.5*sz
        anno_box[:,3][anno_box[:,3]>W]=W
        anno_box[anno_box<0]=0.0
        anno_box=anno_box/np.array([H,W,H,W])
        anno_box=anno_box.tolist()
        # self.anns[seq_name][fid]['bboxes']=box

        if self.data_augmentation == True:
            if random_factor == 0:  # scaling + random color jittering
                minx = 1
                miny = 1
                maxx = 0
                maxy = 0
                for box in anno_box:
                    y1, x1, y2, x2 = box
                    minx = min(minx, x1)
                    miny = min(miny, y1)
                    maxx = max(maxx, x2)
                    maxy = max(maxy, y2)
                maxx = 1 - maxx
                maxy = 1 - maxy
                lim = min(minx, miny, maxx, maxy) * 0.95
                lower = max((1 - lim), 0.7)
                upper = min((1 + lim), 1.3)
                rnd = np.random.randint(0, 100) * (upper - lower) * 0.01 + lower  # get a random number in [lower, upper]
                if rnd <= 1:  # zoom in and crop
                    lim = 1 - rnd
                    i = lim * img.size[0]  # random value to ensure not being cut
                    j = lim * img.size[1]
                    img = crop(img, j, i, img.size[1] - 2 * j, img.size[0] - 2 * i)
                    for box in anno_box:
                        y1, x1, y2, x2 = box
                        y1 = (y1 - 0.5) * 0.5 / (0.5 - lim) + 0.5
                        x1 = (x1 - 0.5) * 0.5 / (0.5 - lim) + 0.5
                        y2 = (y2 - 0.5) * 0.5 / (0.5 - lim) + 0.5
                        x2 = (x2 - 0.5) * 0.5 / (0.5 - lim) + 0.5
                        w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                        temp_boxes.append((w1, h1, w2, h2))
                else:  # zoom out and interpolation
                    lim = rnd - 1
                    img = Pad((int((rnd - 1) * img.size[0]), int((rnd - 1) * img.size[1])), fill=0, padding_mode='reflect')(
                        img)
                    for box in anno_box:
                        y1, x1, y2, x2 = box
                        y1 = (y1 - 0.5) * 0.5 / (0.5 + lim) + 0.5
                        x1 = (x1 - 0.5) * 0.5 / (0.5 + lim) + 0.5
                        y2 = (y2 - 0.5) * 0.5 / (0.5 + lim) + 0.5
                        x2 = (x2 - 0.5) * 0.5 / (0.5 + lim) + 0.5
                        w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                        temp_boxes.append((w1, h1, w2, h2))
                img = ColorJitter(brightness=0.4, contrast=0.25, saturation=0.2)(img)

            elif random_factor == 1:  # horizontal flipping + scaling + random color jittering
                minx = 1
                miny = 1
                maxx = 0
                maxy = 0
                for box in anno_box:
                    y1, x1, y2, x2 = box
                    minx = min(minx, x1)
                    miny = min(miny, y1)
                    maxx = max(maxx, x2)
                    maxy = max(maxy, y2)
                maxx = 1 - maxx
                maxy = 1 - maxy
                lim = min(minx, miny, maxx, maxy) * 0.95
                lower = max((1 - lim), 0.7)
                upper = min((1 + lim), 1.3)
                rnd = np.random.randint(0, 100) * (upper - lower) * 0.01 + lower
                if rnd <= 1:  # zoom in and crop
                    lim = 1 - rnd
                    i = lim * img.size[0]
                    j = lim * img.size[1]
                    img = crop(img, j, i, img.size[1] - 2 * j, img.size[0] - 2 * i)
                    for box in anno_box:
                        y1, x1, y2, x2 = box
                        tmp1 = x1
                        tmp2 = x2
                        x1 = 1 - tmp2
                        x2 = 1 - tmp1
                        y1 = (y1 - 0.5) * 0.5 / (0.5 - lim) + 0.5
                        x1 = (x1 - 0.5) * 0.5 / (0.5 - lim) + 0.5
                        y2 = (y2 - 0.5) * 0.5 / (0.5 - lim) + 0.5
                        x2 = (x2 - 0.5) * 0.5 / (0.5 - lim) + 0.5
                        w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                        temp_boxes.append((w1, h1, w2, h2))
                else:  # zoom out and interpolation
                    lim = rnd - 1
                    img = Pad((int((rnd - 1) * img.size[0]), int((rnd - 1) * img.size[1])), fill=0, padding_mode='reflect')(
                        img)
                    for box in anno_box:
                        y1, x1, y2, x2 = box
                        y1 = (y1 - 0.5) * 0.5 / (0.5 + lim) + 0.5
                        x1 = (x1 - 0.5) * 0.5 / (0.5 + lim) + 0.5
                        y2 = (y2 - 0.5) * 0.5 / (0.5 + lim) + 0.5
                        x2 = (x2 - 0.5) * 0.5 / (0.5 + lim) + 0.5
                        tmp1 = x1
                        tmp2 = x2
                        x1 = 1 - tmp2
                        x2 = 1 - tmp1
                        w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                        temp_boxes.append((w1, h1, w2, h2))
                img = ColorJitter(brightness=0.4, contrast=0.25, saturation=0.2)(img)
                img = transforms.RandomHorizontalFlip(p=1)(img)
        else:
            for box in anno_box:
                y1, x1, y2, x2 = box
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                temp_boxes.append((w1, h1, w2, h2))

        img = transforms.functional.resize(img, self.image_size)
        img = np.array(img)

        # H,W,3 -> 3,H,W
        img = img.transpose(2, 0, 1)
        images.append(img)

        temp_actions = self.anns[seq_name][fid]['actions'][:]
        temp_interactions = self.anns[seq_name][fid]['interactions'][:]
        bboxes_num.append(len(temp_boxes))

        # align the input data
        while len(temp_boxes) != self.num_boxes:
            temp_boxes.append((0, 0, 0, 0))
            temp_actions.append(-1)

        while len(temp_interactions) != self.num_boxes * (self.num_boxes - 1):
            temp_interactions.append(-1)

        bboxes.append(temp_boxes)
        actions.append(temp_actions)
        interactions.append(temp_interactions)

        images = np.stack(images)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes, dtype=np.float).reshape(-1, self.num_boxes, 4)
        actions = np.array(actions, dtype=np.int32).reshape(-1, self.num_boxes)
        interactions = np.array(interactions, dtype=np.int32).reshape(-1, self.num_boxes * (self.num_boxes - 1))

        # convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        interactions = torch.from_numpy(interactions).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()

        return images, bboxes, actions, bboxes_num, interactions, seq_name, fid

def build_tvhi(cfg):
    trn_seq, tst_seq=get_train_test_seq()
    train_anns = TVHI_read_dataset(trn_seq)  # a dict
    train_frames = TVHI_all_frames(train_anns)  # [(seq_id, frm_id)]

    test_anns = TVHI_read_dataset(tst_seq)
    test_frames = TVHI_all_frames(test_anns)  # [(seq_id, frm_id)]

    # build train and test sets
    training_set = TVHIDataset(train_anns,
                              train_frames,
                              cfg.image_size,
                              cfg.out_size,
                              num_boxes=cfg.num_boxes,
                              num_frames=cfg.num_frames,
                              data_augmentation=True)

    validation_set = TVHIDataset(test_anns,
                                test_frames,
                                cfg.image_size,
                                cfg.out_size,
                                num_boxes=cfg.num_boxes,
                                num_frames=cfg.num_frames,
                                data_augmentation=False)

    training_loader=data.DataLoader(training_set,
                                     batch_size=cfg.batch_size,
                                     shuffle=True,
                                     num_workers=cfg.num_workers)

    validation_loader=data.DataLoader(validation_set,
                                      batch_size=cfg.test_batch_size,
                                      shuffle=True,
                                      num_workers=cfg.num_workers)

    print('Build dataset finished...')
    print('%d train samples' % len(train_frames))
    print('%d test samples' % len(test_frames))

    return training_loader,validation_loader

if __name__=='__main__':
    path=data_path
    from config import Config
    from dataset import return_dataset
    import torch
    cfg=Config('tvhi')
    cfg.train_seqs,cfg.test_seqs=get_train_test_seq()
    cfg.data_path=path
    cfg.image_size=540,960
    cfg.out_size=65,117
    cfg.num_boxes=10
    cfg.num_actions=5
    cfg.num_frames=1
    cfg.training_stage=1
    train,test=return_dataset(cfg)
    act_wt=torch.zeros((cfg.num_actions))
    inter_wt=torch.zeros((2))
    for i in train:
        sample=i
        N=sample[3]
        # for a in cfg.num_actions:
        #     act_wt[a]+=torch.sum(sample[2][0][0:N]==a)
        # for r in 2:
        #     inter_wt[r]+=torch.sum(sample[4][0][0:N*(N-1)]==r)
        for a in cfg.num_actions:
            act_wt[a]+=torch.sum(sample[2][0][0:N]==a)
        for r in 2:
            inter_wt[r]+=torch.sum(sample[4][0][0:N*(N-1)]==r)
        # act_wt[sample[2][0][0:N]]+=1
        # inter_wt[sample[4][0][0:N*(N-1)]] += 1
    print(act_wt,inter_wt)
    print(max(act_wt)/act_wt,max(inter_wt)/inter_wt)
    # for i in test:
    #     print(i[0].shape)
        # print(d)
        # input()











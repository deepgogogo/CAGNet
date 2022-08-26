from torch.utils import data
import torchvision.transforms as transforms
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *

# import random
from PIL import Image
import numpy as np
import re
import os

FRAMES_SIZE=(240,320) # (height, width), before resize by backbone
action_dict={'bend':0,'box':1,'handshake':2,'hifive':3,'hug':4,'kick':5,'pat':6,'push':7,'others':8,
             'be_bended':8, 'be_boxed':8, 'be_kicked':8, 'be_pated':8, 'be_pushed':8, 'no_action':8} # all action classes

action_dict_asym={'bend':0,'box':1,'handshake':2,'hifive':3,'hug':4,'kick':5,'pat':6,'push':7,'others':8,
             'be_bended':9, 'be_boxed':10, 'be_kicked':11, 'be_pated':12, 'be_pushed':13, 'no_action':8} # all action classes

# CAGNet/data/BIT
data_path=os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        'BIT')

def get_train_test_seq():
    clsNames = [cls for cls in action_dict.keys() if cls not in ['be_bended','be_boxed','be_kicked','be_pated','be_pushed',
                                                             'others','no_action']]
    trn_seq = ['{}_{:04d}'.format(i, j) for i in clsNames for j in range(1,35)] # first 34 used as train set
    tst_seq = ['{}_{:04d}'.format(i, j) for i in clsNames for j in range(35, 51)] # rest 16 used as test set
    return trn_seq, tst_seq

def get_train_test_seq_from_partition():
    trn_seq=[]
    tst_seq=[]
    partition_path=os.path.join(data_path,'partition.txt')
    with open(partition_path,'r')as f:
        for line in f.readlines():
            g=re.split(' +',line.strip())
            if g[1]=='train':trn_seq.append(g[0])
            if g[1]=='test':tst_seq.append(g[0])
    return trn_seq,tst_seq

###########################
#
# read annotations from an annotation file. All frames without interactions are discarded
#

def BIT_read_annotations(sname):# path to all seqs, seq name for one video
    annotations = {}
    cls = sname.split('_')[0] # class name
    path = data_path + '/BIT-anno/tidy_anno/' + cls + '/' + sname+'.txt'
    frame_id = 0
    box_num = 0
    H,W = FRAMES_SIZE

    with open(path,mode='r',encoding='utf–8') as f:
        box_num = 0

        for l in f.readlines():
            values = l.split()
            if len(values) == 4: # frame id and box number
                action_names = []
                frame_id=int(values[1])
                box_num=int(values[3])
                index=0
                annotations[frame_id] = {
                        'frame_id':frame_id,
                        'actions':[], # eg. [4, 4, 4, 4]
                        'interactions':[0 for i in range(box_num*(box_num-1))],
                        'box_num':box_num,
                        'bboxes':[] # normalized coordinates of top-left and bottom-right corners, eg.[(a1,b1,c1,d1), ..., (a4,b4,c4,d4)]
                }
            else: #action
                if values[5] == 'highfive':
                    action_names.append('hifive')
                else:
                    action_names.append(values[5])
                annotations[frame_id]['actions'].append(action_dict_asym[action_names[-1]])
                x1,y1,x2,y2 = (int(values[i])  for i  in range(1,5)) # box_W = x2 - x1, box_H = y2 - y1
                annotations[frame_id]['bboxes'].append((y1/H,x1/W,y2/H,x2/W)) # y_TL, x_TL, y_BR, x_BR
                index += 1
            if index == box_num: # reading info for a frame finished
                acts = annotations[frame_id]['actions']
                cc = 0
                for i in range(box_num):
                    for j in range(box_num):
                        if i == j:
                            continue
                        if (acts[i] == acts[j] and acts[i] != 8) or (acts[i] > 8 and acts[j] < 8) or (acts[j] > 8 and acts[i] < 8):
                            annotations[frame_id]['interactions'][cc] = 1
                        cc += 1
                annotations[frame_id]['actions'] = [action_dict[c] for c in action_names] # transfer "be_..." classes into "no_action"
                index = 0

        #remove frames box_num le 1
        # for k, v in list(annotations.items()):
        #     if v['box_num'] <= 1:
        #         annotations.pop(k)

        # remove frames without interaction
        for k, v in list(annotations.items()):
            if sum(v['interactions']) == 0:
                annotations.pop(k)
    return annotations

def BIT_read_dataset(seqs):
    data = {}
    for sname in seqs: # sname: seq name, for example bend_0001
        data[sname] = BIT_read_annotations(sname) # each data[sname] is a dict with frm id as keys
    return data

def BIT_all_frames(anns): 
    return [(s,f) for s in anns for f in anns[s]] # (s:seq name, f:frame id)

class BITDataset(data.Dataset): # create a dataset
    def __init__(self,anns,frames,image_size,feature_size,num_boxes=5,num_frames=1,
                 data_augmentation=False):
        '''anns: return by BIT_read_dataset
           frames: return by BIT_all_frames
           data_path: BIT data path
           feature_size: the output feature map size of backbone
        '''
        self.anns = anns
        self.frames = frames
        self.image_path = data_path+'/Bit-frames'
        self.image_size = image_size
        self.feature_size = feature_size
        
        self.num_boxes = num_boxes
        self.num_frames = num_frames
        
        self.data_augmentation = data_augmentation
    
    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """
        sample = self.load_samples_sequence(self.frames[index])
        return sample

    def load_samples_sequence(self,select_frame):
        # print(select_frame)    # for debug only
        random_factor = np.random.randint(0, 2)
        OH, OW = self.feature_size
        images, bboxes = [], []
        actions = []
        interactions = []
        bboxes_num = []

        seq_name, fid = select_frame
        name_id = seq_name.split('_')
        sid = int(name_id[1])
        scls = name_id[0]
        img = Image.open(self.image_path + '/' + scls + '/' + seq_name + '/' + '{:04d}.jpg'.format(fid))

        temp_boxes = []
        if self.data_augmentation == True:
            if random_factor == 0:# scaling + random color jittering
                minx = 1
                miny = 1
                maxx = 0
                maxy = 0
                for box in self.anns[seq_name][fid]['bboxes']:
                    y1,x1,y2,x2 = box
                    minx = min(minx,x1)
                    miny = min(miny,y1)
                    maxx = max(maxx,x2)
                    maxy = max(maxy,y2)
                maxx = 1 - maxx
                maxy = 1 - maxy
                lim = min(minx,miny,maxx,maxy) * 0.95
                lower = max((1-lim),0.7)
                upper = min((1+lim),1.3)
                rnd = np.random.randint(0,100)*(upper-lower)*0.01+lower # get a random number in [lower, upper]
                if rnd <= 1:# zoom in and crop
                    lim = 1 - rnd
                    i = lim*img.size[0] # random value to ensure not being cut
                    j = lim*img.size[1]
                    img = crop(img, j, i, img.size[1]-2*j, img.size[0]-2*i)
                    for box in self.anns[seq_name][fid]['bboxes']:
                        y1,x1,y2,x2 = box
                        y1 = (y1-0.5)*0.5/(0.5-lim)+0.5
                        x1 = (x1-0.5)*0.5/(0.5-lim)+0.5
                        y2 = (y2-0.5)*0.5/(0.5-lim)+0.5
                        x2 = (x2-0.5)*0.5/(0.5-lim)+0.5
                        w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                        temp_boxes.append((w1,h1,w2,h2))
                else: # zoom out and interpolation
                    lim = rnd - 1
                    img = Pad((int((rnd-1)*img.size[0]),int((rnd-1)*img.size[1])), fill=0, padding_mode='reflect')(img)
                    for box in self.anns[seq_name][fid]['bboxes']:
                        y1,x1,y2,x2 = box
                        y1 = (y1-0.5)*0.5/(0.5+lim) + 0.5
                        x1 = (x1-0.5)*0.5/(0.5+lim) + 0.5
                        y2 = (y2-0.5)*0.5/(0.5+lim) + 0.5
                        x2 = (x2-0.5)*0.5/(0.5+lim) + 0.5
                        w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH
                        temp_boxes.append((w1,h1,w2,h2))
                img = ColorJitter(brightness = 0.4, contrast=0.25, saturation = 0.2)(img)

            elif random_factor == 1: # horizontal flipping + scaling + random color jittering
                minx = 1
                miny = 1
                maxx = 0
                maxy = 0
                for box in self.anns[seq_name][fid]['bboxes']:
                    y1,x1,y2,x2 = box
                    minx = min(minx,x1)
                    miny = min(miny,y1)
                    maxx = max(maxx,x2)
                    maxy = max(maxy,y2)
                maxx = 1-maxx
                maxy = 1-maxy
                lim = min(minx,miny,maxx,maxy)*0.95
                lower = max((1-lim),0.7)
                upper = min((1+lim),1.3)
                rnd = np.random.randint(0,100)*(upper-lower)*0.01+lower
                if rnd <= 1: # zoom in and crop
                    lim = 1 - rnd
                    i = lim*img.size[0]
                    j = lim*img.size[1]
                    img = crop(img, j, i, img.size[1]-2*j, img.size[0]-2*i)
                    for box in self.anns[seq_name][fid]['bboxes']:
                        y1,x1,y2,x2 = box
                        tmp1 = x1
                        tmp2 = x2
                        x1 = 1-tmp2
                        x2 = 1-tmp1
                        y1 = (y1-0.5)*0.5/(0.5-lim)+0.5
                        x1 = (x1-0.5)*0.5/(0.5-lim)+0.5
                        y2 = (y2-0.5)*0.5/(0.5-lim)+0.5
                        x2 = (x2-0.5)*0.5/(0.5-lim)+0.5
                        w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                        temp_boxes.append((w1,h1,w2,h2))
                else: # zoom out and interpolation
                    lim=rnd-1
                    img = Pad((int((rnd-1)*img.size[0]),int((rnd-1)*img.size[1])), fill=0, padding_mode='reflect')(img)
                    for box in self.anns[seq_name][fid]['bboxes']:
                        y1,x1,y2,x2 = box
                        y1 = (y1-0.5)*0.5/(0.5+lim)+0.5
                        x1 = (x1-0.5)*0.5/(0.5+lim)+0.5
                        y2 = (y2-0.5)*0.5/(0.5+lim)+0.5
                        x2 = (x2-0.5)*0.5/(0.5+lim)+0.5
                        tmp1 = x1
                        tmp2 = x2
                        x1 = 1-tmp2
                        x2 = 1-tmp1
                        w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH
                        temp_boxes.append((w1,h1,w2,h2))
                img = ColorJitter(brightness=0.4,contrast=0.25, saturation=0.2)(img)
                img = transforms.RandomHorizontalFlip(p=1)(img)
        else:
            for box in self.anns[seq_name][fid]['bboxes']:
                y1,x1,y2,x2=box
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH 
                temp_boxes.append((w1,h1,w2,h2))

        img = transforms.functional.resize(img,self.image_size)
        img = np.array(img)

        # H,W,3 -> 3,H,W
        img = img.transpose(2,0,1)
        images.append(img)
                
        temp_actions = self.anns[seq_name][fid]['actions'][:]
        temp_interactions = self.anns[seq_name][fid]['interactions'][:]
        bboxes_num.append(len(temp_boxes))

        # align the input data
        while len(temp_boxes) != self.num_boxes:
            temp_boxes.append((0,0,0,0))
            temp_actions.append(-1)

        while len(temp_interactions) != self.num_boxes*(self.num_boxes-1):
            temp_interactions.append(-1)

        bboxes.append(temp_boxes)
        actions.append(temp_actions)
        interactions.append(temp_interactions)
        
        images = np.stack(images)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes,dtype=np.float).reshape(-1,self.num_boxes,4)
        actions = np.array(actions,dtype=np.int32).reshape(-1,self.num_boxes)
        interactions = np.array(interactions,dtype=np.int32).reshape(-1,self.num_boxes*(self.num_boxes-1))
        
        #convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        interactions = torch.from_numpy(interactions).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        
        return images, bboxes, actions, bboxes_num, interactions, seq_name, fid

def build_bit(cfg):
    trn_seq, tst_seq=get_train_test_seq()
    train_anns = BIT_read_dataset(trn_seq)  # a dict
    train_frames = BIT_all_frames(train_anns)  # [(seq_id, frm_id)]

    test_anns = BIT_read_dataset(tst_seq)
    test_frames = BIT_all_frames(test_anns)  # [(seq_id, frm_id)]

    # build train and test sets
    training_set = BITDataset(train_anns,
                              train_frames,
                              cfg.image_size,
                              cfg.out_size,
                              num_boxes=cfg.num_boxes,
                              num_frames=cfg.num_frames,
                              data_augmentation=True)

    validation_set = BITDataset(test_anns,
                                test_frames,
                                cfg.image_size,
                                cfg.out_size,
                                num_boxes=cfg.num_boxes,
                                num_frames=cfg.num_frames,
                                data_augmentation=False)

    training_loader=data.DataLoader(training_set,
                                     batch_size=cfg.batch_size,
                                     shuffle=True,
                                     num_workers=cfg.num_workers,
                                     )

    validation_loader=data.DataLoader(validation_set,
                                      batch_size=cfg.test_batch_size,
                                      shuffle=True,
                                      num_workers=cfg.num_workers,
                                      )

    print('Build dataset finished...')
    print('%d train samples' % len(train_frames))
    print('%d test samples' % len(test_frames))

    return training_loader,validation_loader

if __name__ == "__main__":
    A = BIT_read_annotations('BIT', 'kick_0035')
    print(A)

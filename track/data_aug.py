import random
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import Pad
from torchvision.transforms.functional import crop,resize
import copy
from typing import Tuple, List, Optional
import numbers
from torch import Tensor
import numpy as np

    
class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.25, saturatio=0.2, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturatio, hue)

    def __call__(self, samples):
        frames=samples['frames']
        out_frames=[]
        for img in frames:
            out_frames.append(self.color_jitter(img))
        samples['frames']=out_frames
        return samples
    
    
    
def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes
    return flipped_image, target

class RandomResizeBoxes(object):
    """randomly resize box size"""
    def __init__(self, p=0.5):
        self.p=p
        
    def random_resize_boxes(self, img, boxes):
        temp_boxes=[]
        
        minx = 1
        miny = 1
        maxx = 0
        maxy = 0
        for box in boxes:
            # y1,x1,y2,x2 = box
            x1,y1,x2,y2 = box
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
            out_img = crop(img, j, i, img.size[1]-2*j, img.size[0]-2*i)
            for box in boxes:
                x1,y1,x2,y2 = box
                y1 = (y1-0.5)*0.5/(0.5-lim)+0.5
                x1 = (x1-0.5)*0.5/(0.5-lim)+0.5
                y2 = (y2-0.5)*0.5/(0.5-lim)+0.5
                x2 = (x2-0.5)*0.5/(0.5-lim)+0.5
                temp_boxes.append([x1,y1,x2,y2])
        else: # zoom out and interpolation
            lim = rnd - 1
            out_img = Pad((int((rnd-1)*img.size[0]),int((rnd-1)*img.size[1])), fill=0, padding_mode='reflect')(img)
            for box in boxes:
                x1,y1,x2,y2 = box
                y1 = (y1-0.5)*0.5/(0.5+lim) + 0.5
                x1 = (x1-0.5)*0.5/(0.5+lim) + 0.5
                y2 = (y2-0.5)*0.5/(0.5+lim) + 0.5
                x2 = (x2-0.5)*0.5/(0.5+lim) + 0.5
                temp_boxes.append([x1,y1,x2,y2])
        assert len(temp_boxes)!=0
        return out_img,temp_boxes

    def __call__(self, samples):
        frames=samples['frames']
        boxes=samples['bboxes']
        out_frames,out_boxes=[],[]
        
        for img, box in zip(frames,boxes):
            if random.random()<self.p:
                oimg,obox=self.random_resize_boxes(img,box)
                out_frames.append(oimg)
                out_boxes.append(obox)
            else:
                out_frames.append(img)
                out_boxes.append(box)
            
        samples['frames']=out_frames
        samples['bboxes']=out_boxes
        return samples
                
class InstancePadding(object):
    """pad instance num (including boxes,label etc) to maxn for gpu batch loading"""
    def __init__(self,max_n):
        self.n=max_n
        
    def __call__(self, samples):
        actions=samples['actions']
        interactions=samples['interactions']
        box_nums=samples['box_nums']
        bboxes=samples['bboxes']
        track_ids=samples['track_ids']
        
        for action,bbox,track_id,interaction,n in zip(actions,bboxes,track_ids,interactions,box_nums):
            assert n<=self.n
            # align the input data
            while len(bbox) < self.n:
                bbox.append((0,0,0,0))
                action.append(-1)
                track_id.append(-1)

            while len(interaction) < self.n*(self.n-1):
                interaction.append(-1)

        return samples
        
            
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, samples):
        frames=samples['frames']
        bboxes=samples['bboxes']
        out_frames,out_bboxes=[],[]
        if random.random() < self.p:
            for img,box in zip(frames,bboxes):
                out_frames.append(F.hflip(img))
                out_box=np.array(box)[:,[2,1,0,3]]*np.array([-1, 1, -1, 1])+np.array([1,0,1,0])
                out_bboxes.append(out_box.tolist())
        
            samples['frames']=out_frames
            samples['bboxes']=out_bboxes
        return samples
    
    
class ResizePIL(object):
    def __init__(self,img_sz):
        self.img_sz=img_sz
        pass
    
    def __call__(self,samples):
        frames=samples['frames']
        out_frames=[]
        for img in frames:
            out_frames.append(resize(img,self.img_sz))
        samples['frames']=out_frames
        return samples
    
class PIL2Numpy(object):
    
    def __init__(self):
        pass
    
    def __call__(self, samples):
        frames=samples['frames']
        out_frames=[]
        for img in frames:
            out_frames.append(np.array(img).transpose(2,0,1))
        samples['frames']=out_frames
        return samples
        
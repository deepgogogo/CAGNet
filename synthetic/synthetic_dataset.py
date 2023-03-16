from torch.utils.data import Dataset,DataLoader,DistributedSampler
import torch.distributed as dist
from pathlib import Path
import pickle
import numpy as np

class Synthetics(Dataset):
    
    def __init__(self,data_path):
        data_path=Path(data_path)
        with open(data_path,'rb') as f:
            bytes=pickle.load(f)
            self.score=bytes['score']
            self.label=bytes['label']
        self.conflict_matrix=np.load(data_path.parent/'action_grid.npy')
        assert len(self.score)==len(self.label)
        
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,index):
        return self.score[index],self.label[index]
    
def build_synthetics():
    batch_size=1
    world_size=dist.get_world_size()
    rank=dist.get_rank()
    
    train_set=Synthetics('synthetic_data/train')
    test_set=Synthetics('synthetic_data/test')
    
    train_sampler=DistributedSampler(train_set,num_replicas=world_size,
                                     rank=rank,shuffle=True,drop_last=False)
    test_sampler=DistributedSampler(test_set,num_replicas=world_size,
                                     rank=rank,shuffle=False,drop_last=False)
    
    train_loader=DataLoader(train_set,batch_size=batch_size,sampler=train_sampler)
    test_loader=DataLoader(test_set,batch_size=batch_size,sampler=test_sampler)
    
    return train_loader,test_loader

if __name__=='__main__':
    train_loader,test_loader=build_synthetics()
    for sample in train_loader:
        print(sample)
    pass
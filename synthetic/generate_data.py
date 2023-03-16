import numpy as np
import pickle
import json
import uuid
import argparse
from sklearn.metrics import  precision_recall_fscore_support,confusion_matrix, accuracy_score
from scipy.special import softmax
from itertools import combinations,permutations
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import random

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

def generate_data_demo():
    """generate action scores and interaction scores 
    mimicing the outputs of basemodel,
    """
    # action classes
    # conflict matrix, a. symetric; b. asymetric
    # predefined confusion matrix of action classes mimicing the  similar semantics of different action classes
    
    ## predefined
    np.random.seed(0)
    N_ACT=5 # num of action classes, supposed to be large 
    N_SAMP=10 # num of all data samples, i.e num of images
    N_PER=10 # max num of person per image
    
    ## generation
    conflict_matrix=1-np.eye(N_ACT,N_ACT)
    
    action_classes=np.arange(N_ACT)
    # person_num=np.random.randint(1,N_PER+1,size=(N_SAMP))

    ### generate ground truth labels
    person_num=np.ones((N_SAMP,))*2
    action_labels=np.random.randint(0,N_ACT,(N_SAMP,))
    
    ### generate predicted scores
    action_scores=np.random.randn(N_SAMP, N_ACT)
    ratio=0.75
    chosen=np.random.choice(np.arange(N_SAMP),int(N_SAMP*ratio),replace=False)
    # action_scores[chosen,action_labels[chosen]]=5. # possibly fail
    action_scores[chosen,action_labels[chosen]]=5.+np.random.randn(*chosen.shape)
    
    action_scores=softmax(action_scores,axis=-1)
    
    ### verify
    action_pred_label=np.argmax(action_scores,axis=-1)
    accuracy=accuracy_score(action_labels,action_pred_label)
    actions_precision, actions_recall, actions_F1, support = precision_recall_fscore_support(action_labels,action_pred_label,beta=1, average='macro')
    conf_mat=confusion_matrix(action_labels,action_pred_label,normalize='true')
    # conf_mat = conf_mat/np.sum(conf_mat, axis=-1,keepdims=True)
    
    ## persistency
    pass

def generate_data_demo1(dataset_size=5000):
    
    np.random.seed(0)
    N_ACT=50 # num of action classes, supposed to be large 
    N_SAMP=10 # num of all data samples, i.e num of images
    N_PER=10 # max num of person per image
    conflict_matrix=1-np.eye(N_ACT,N_ACT)
    
    def chosen(p=0.75):
        return np.random.rand(1)<p
    
    act_labels,act_scores=[],[]
    peak=5.
    for id in range(dataset_size):
        
        num_person=2
        act_score=np.random.randn(num_person,N_ACT)
        act_label=np.random.randint(0,N_ACT,num_person)
        if chosen():
            act_label[:]=np.random.randint(0,N_ACT)
            act_score[np.arange(num_person),act_label]=peak+np.random.randn(num_person)
        act_labels.append(act_label)
        act_scores.append(act_score)
    
    act_scores=softmax(np.array(act_scores),axis=-1)
    act_pred_labels=np.argmax(act_scores,axis=-1)
    act_labels=np.array(act_labels)
    accuracy=accuracy_score(act_labels.flatten(),act_pred_labels.flatten())
    actions_precision, actions_recall, actions_F1, support = precision_recall_fscore_support(act_labels.flatten(),act_pred_labels.flatten(),beta=1, average='macro')
    print(f'accuray={accuracy}\nprecision={actions_precision}\nrecall={actions_recall}\nF1={actions_F1}\n')
    
    pass

def make_action_grid(n_act=100,n_diag=50):
    assert n_act >= n_diag
    # 0 denotes compatible action 
    grid=np.ones((n_act,n_act),dtype=int)
    assigned=np.zeros(n_act,dtype=int)
    diag=np.random.choice(np.arange(n_act-1),n_diag,replace=False)
    assigned[diag]=1
    grid[diag,diag]=0
    assigned[n_act-1]=1 # no action 
    group_sizes=np.array([2,3,4,5]) # num of different action labels in one group
    # gs_prob=[0.4,0.3,0.2,0.1]
    gs_prob=[0.25,0.25,0.25,0.25]
    group_cnt=np.zeros(6)
    
    while (assigned==0).any():
        remain=(assigned==0).nonzero()[0]
        gs=np.random.choice(group_sizes,1,p=gs_prob)
        if gs<=len(remain):
            group_cnt[gs]+=1
            if gs+1==n_act-assigned.sum() and gs!=len(remain): # avoid single action
                gs+=1
            chosen=np.random.choice(remain,gs,replace=False)
            print(chosen.tolist())
            idx=np.array(list(permutations(chosen,2)))
            grid[idx[:,0],idx[:,1]]=0
            assigned[chosen]=1
    np.save('synthetic_data/action_grid',grid)
    print(group_cnt)
    pass

def draw_action_grid(grid_path='action_grid.npy', save_name='action_grid.pdf'):
    dir=Path(__file__).resolve().parent/'synthetic_data'
    grid=1-np.load(dir/grid_path)
    plt.imshow(grid,cmap=plt.get_cmap('summer'))
    plt.savefig(dir/save_name)
    
def generate_data(dataset_size=20000,train_size=12000,n_act=100):
    np.random.seed(0)
    no_act_label=n_act-1
    conflict_matrix=np.load('synthetic_data/action_grid.npy')
    assert conflict_matrix.shape[0]==n_act
    flip_matrix=1-conflict_matrix
    
    # generate labels
    dataset=[]
    tail_p=[]
    p=1
    for _ in range(4):
        tail_p.extend([p]*5)
        p/=2
    for _ in range(79):
        p-=0.0005
        tail_p.append(p)
    tail_p=tail_p/np.sum(tail_p)
    
    for i in range(dataset_size):
        n_inter=np.random.choice(np.arange(4),1,p=np.array([0.01,0.33,0.33,0.33]))
        img_ground_truth={'n_inter':n_inter.item()}
        
        if n_inter.item()==0:
            n_person=np.random.choice(np.arange(2,11),1).item()
            act_label=np.full(n_person,no_act_label,dtype=int)
            pair_inter_label=np.full((n_person,n_person),0,dtype=int)
            img_ground_truth['n_person']=n_person
            img_ground_truth['interaction']=[{'person_id':np.arange(n_person,dtype=int),'group_act_label':np.full(n_person,no_act_label,dtype=int)}]
            img_ground_truth['pair_inter_label']=pair_inter_label
            img_ground_truth['act_label']=act_label
        else:
            act_label=np.random.choice(np.arange(n_act-1),n_inter,p=tail_p)
            group_sz=flip_matrix[act_label].sum(axis=1)+1
            n_person=group_sz.sum()
            img_ground_truth['n_person']=n_person
            unchosen=np.full(n_person,True,dtype=bool)
            pair_inter_label=np.full((n_person,n_person),0,dtype=int)
            interaction=[]
            
            all_act_label=[]
            all_person_id=[]
            
            for g in range(n_inter.item()):
                person_id=np.random.choice(np.arange(n_person)[unchosen],group_sz[g],replace=False)
                all_person_id.append(person_id)
                unchosen[person_id]=False
                idx=np.array(list(permutations(person_id,2)))
                pair_inter_label[idx[:,0],idx[:,1]]=1
                group_act_label=flip_matrix[act_label[g]].nonzero()[0]
                group_act_label=np.concatenate((np.atleast_1d(act_label[g]),group_act_label))
                all_act_label.append(group_act_label)
                interaction.append({'person_id':person_id,'group_act_label':group_act_label})
                print(f'person_id={person_id},group act label={group_act_label}')
            
            all_act_label=np.concatenate(all_act_label)
            all_person_id=np.concatenate(all_person_id)
            
            img_ground_truth['act_label']=all_act_label[np.argsort(all_person_id)]
            img_ground_truth['interaction']=interaction
            img_ground_truth['pair_inter_label']=pair_inter_label
        dataset.append(img_ground_truth)
        
    # generate scores
    peak=5.
    scores=[]
    for anno in dataset:
        n_inter=anno['n_inter']
        n_person=anno['n_person']
        # pair_inter_score=np.full((n_person,n_person),0,dtype=float)
        pair_inter_label=anno['pair_inter_label']
        pair_inter_score=softmax(np.random.randn(n_person,n_person,2),axis=-1)
        chosen=np.random.choice([0,1],(n_person,n_person),p=[0.25,0.75]).nonzero()
        pair_inter_score[chosen[0],chosen[1],pair_inter_label[chosen[0],chosen[1]]]=peak+np.random.randn()
        act_score=[]
        if anno['n_inter']==0:
            logits=np.random.randn(n_person,n_act)
            chosen=np.random.choice([0,1],n_person,p=[0.25,0.75]).nonzero()[0]
            logits[chosen,n_act-1]=peak+np.random.randn()
            logits=softmax(logits,axis=-1)
            act_score.append(logits)
        else:
            person_id,group_act_label=[],[]
            for group in anno['interaction']:
                person_id.append(group['person_id'])
                group_act_label.append(group['group_act_label'])
            person_id=np.concatenate(person_id)
            group_act_label=np.concatenate(group_act_label)
            group_act_label=group_act_label[np.argsort(person_id)]
                
            logits=np.random.randn(n_person,n_act)
            chosen=np.random.choice([0,1],n_person,p=[0.25,0.75]).nonzero()[0]
            logits[chosen,group_act_label[chosen]]=peak+np.random.randn()
            logits=softmax(logits,axis=-1)
            
            for group in anno['interaction']:
                act_score.append(logits[group['person_id']])
            # act_score.append(softmax(logits,axis=-1))
            # act_score.append(softmax(np.random.randn(len(person_id),n_act),axis=-1))
                
        scores.append({'pair_inter_score':pair_inter_score,'group_act_score':act_score,'act_score':logits})
    
    save_path=Path('synthetic_data')
    save_path.mkdir(exist_ok=True)
    
    with open(save_path/'train','wb') as f:
        pickle.dump({'score':scores[:train_size],'label':dataset[:train_size]},f)
    with open(save_path/'test','wb') as f:
        pickle.dump({'score':scores[train_size:],'label':dataset[train_size:]},f)
        
    # verify
    act_groundtruth=[]
    act_predict=[]
    inter_groundtruth=[]
    inter_predict=[]
    for score, target in zip(scores,dataset):
        
        group_act_predict,group_act_label,person_id=[],[],[]
        
        for act_scr, group_info in zip(score['group_act_score'],target['interaction']):
            act_pred_label=np.argmax(act_scr,axis=-1)
            group_act_predict.append(act_pred_label)
            person_id.append(group_info['person_id'])
            group_act_label.append(group_info['group_act_label'])
            
        group_act_predict=np.concatenate(group_act_predict)
        person_id=np.concatenate(person_id)
        group_act_label=np.concatenate(group_act_label)
        act_predict.append(group_act_predict[np.argsort(person_id)])
        act_groundtruth.append(group_act_label[np.argsort(person_id)])
            
        n_person=target['n_person']
        assert len(group_act_predict)==n_person
        inter_scr=score['pair_inter_score']
        inter_scr=inter_scr[~np.eye(n_person,n_person,dtype=bool)]
        inter_pred_label=np.argmax(inter_scr,axis=-1)
        inter_label=target['pair_inter_label'][~np.eye(n_person,n_person,dtype=bool)]
        inter_predict.append(inter_pred_label)
        inter_groundtruth.append(inter_label)
        pass
    
    uncompat,untrans=count_conflict(act_predict,inter_predict,conflict_matrix)
    print(f'uncompat={uncompat},untrans={untrans}')
        
    act_predict=np.concatenate(act_predict)
    act_groundtruth=np.concatenate(act_groundtruth)
    accuracy=accuracy_score(act_groundtruth,act_predict)
    actions_precision, actions_recall, actions_F1, support = precision_recall_fscore_support(act_groundtruth,act_predict,beta=1, average='macro')
    print(f'accuray={accuracy}\nprecision={actions_precision}\nrecall={actions_recall}\nF1={actions_F1}\n')
  
    inter_predict=np.concatenate(inter_predict)
    inter_groundtruth=np.concatenate(inter_groundtruth)
    accuracy=accuracy_score(inter_groundtruth,inter_predict)
    inter_precision, inter_recall, inter_F1, support = precision_recall_fscore_support(inter_groundtruth,inter_predict,beta=1, average='macro')
    print(f'accuray={accuracy}\nprecision={inter_precision}\nrecall={inter_recall}\nF1={inter_F1}\n')
    pass

def load_data(data_path='synthetic_data/test'):
    conflict_matrix=np.load('synthetic_data/action_grid.npy')
    flip_matrix=1-conflict_matrix
    num_actions=len(conflict_matrix)

    save_path=Path(data_path)
    with open(save_path,'rb') as f:
        bytes=pickle.load(f)
        scores=bytes['score']
        dataset=bytes['label']
        
    # verify
    act_groundtruth=[]
    act_predict=[]
    inter_groundtruth=[]
    inter_predict=[]
    for score, target in zip(scores,dataset):
        
        group_act_predict,group_act_label,person_id=[],[],[]
        
        for act_scr, group_info in zip(score['group_act_score'],target['interaction']):
            act_pred_label=np.argmax(act_scr,axis=-1)
            group_act_predict.append(act_pred_label)
            person_id.append(group_info['person_id'])
            group_act_label.append(group_info['group_act_label'])
            
        group_act_predict=np.concatenate(group_act_predict)
        person_id=np.concatenate(person_id)
        group_act_label=np.concatenate(group_act_label)
        act_predict.append(group_act_predict[np.argsort(person_id)])
        act_groundtruth.append(group_act_label[np.argsort(person_id)])
            
        n_person=target['n_person']
        assert len(group_act_predict)==n_person
        inter_scr=score['pair_inter_score']
        inter_scr=inter_scr[~np.eye(n_person,n_person,dtype=bool)]
        inter_pred_label=np.argmax(inter_scr,axis=-1)
        inter_label=target['pair_inter_label'][~np.eye(n_person,n_person,dtype=bool)]
        inter_predict.append(inter_pred_label)
        inter_groundtruth.append(inter_label)
        pass
    
    uncompat,untrans=count_conflict(act_predict,inter_predict,conflict_matrix)
    print(f'uncompat={uncompat},untrans={untrans}')
        
    act_predict=np.concatenate(act_predict)
    act_groundtruth=np.concatenate(act_groundtruth)
    act_accuracy=accuracy_score(act_groundtruth,act_predict)
    actions_precision, actions_recall, actions_F1, support = precision_recall_fscore_support(act_groundtruth,act_predict,beta=1, average='macro')
    print(f'act:accuray={act_accuracy}\nprecision={actions_precision}\nrecall={actions_recall}\nF1={actions_F1}\n')
    act_f1=precision_recall_fscore_support(act_groundtruth,act_predict,beta=1, average=None)[2]
    print(f'f1 scores for each act {act_f1.tolist()}')
  
    inter_predict=np.concatenate(inter_predict)
    inter_groundtruth=np.concatenate(inter_groundtruth)
    inter_accuracy=accuracy_score(inter_groundtruth,inter_predict)
    inter_precision, inter_recall, inter_F1, support = precision_recall_fscore_support(inter_groundtruth,inter_predict,beta=1, average='macro')
    print(f'inter:accuray={inter_accuracy}\nprecision={inter_precision}\nrecall={inter_recall}\nF1={inter_F1}\n')
    
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
    
    accuracy=(act_accuracy+inter_accuracy)/2
    precision=(actions_precision+inter_precision)/2
    recall=(actions_recall+inter_recall)/2
    F1=(actions_F1+inter_F1)/2
    print(f'accuray={accuracy}\nprecision={precision}\nrecall={recall}\nF1={F1}\n')
    print(f'mean iou={mean_iou}')
    pass 

def draw_action_bar(data_path='synthetic_data/train'):
    
    conflict_matrix=np.load('synthetic_data/action_grid.npy')
    flip_matrix=1-conflict_matrix
    num_actions=len(conflict_matrix)

    save_path=Path(data_path)
    with open(save_path,'rb') as f:
        bytes=pickle.load(f)
        # scores=bytes['score']
        dataset=bytes['label']
    act_cnt=np.zeros(num_actions,dtype=int)
    for target in dataset:
        act_label=target['act_label']
        np.add.at(act_cnt,act_label,1)
        
    fig,ax=plt.subplots(figsize=(15,5))
    ax.bar(np.arange(num_actions),np.sort(act_cnt[:num_actions])[::-1])
    # ax.bar(np.arange(num_actions),act_cnt[:num_actions])
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.subplots_adjust(left=0.9, right=0.9)
    ax.autoscale(axis='x', tight=True)
    ax.autoscale(axis='y', tight=True)
    # ax.set_title("Count of Action")
    plt.savefig('action_bar.jpg')
    plt.savefig('action_bar.pdf')
    pass



if __name__=='__main__':
    # generate_data_demo()
    
    # make_action_grid()
    # draw_action_grid()
    # generate_data()
    # draw_action_bar()
    
    load_data(data_path='synthetic_data/test')
    # action_bar()
    pass
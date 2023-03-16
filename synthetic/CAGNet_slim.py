import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np
from itertools import combinations

from utils import *
from torchvision.ops import *

from meanfield_slim import meanfield
from mpnn import factor_mpnn
from utils import unpackEdgeFeature
import torch.distributed as dist

# base_model_fgnn
# 构造factor graph
def generate_graph(N):
    # n[0,n-1],z[n,n+n*(n-1)/2-1],
    # h[n+n*(n-1)/2,n+n*(n-1)-1],
    # g[n+n*(n-1),n+n*(n-1)+n*(n-1)*(n-2)/6)-1]
    G=[[] for _ in range(N+N*(N-1)+N*(N-1)*(N-2)//6)]
    h_id={ c:i+N for i,c in enumerate(combinations(range(N),2))}#id of z
    hidx=N+N*(N-1)//2
    for u in range(N):
        for v in range(u+1,N):
            G[hidx].extend([u,v,h_id[(u,v)]])
            G[u].append(hidx)
            G[v].append(hidx)
            G[h_id[(u,v)]].append(hidx)
            hidx+=1

    if N>2:
        gidx=N+N*(N-1)
        for i in range(N):
            for j in range(i+1,N):
                for k in range(j+1,N):
                    z1,z2,z3=h_id[(i,j)],h_id[(i,k)],h_id[(j,k)]
                    G[gidx].extend([z1,z2,z3])
                    G[z1].append(gidx)
                    G[z2].append(gidx)
                    G[z3].append(gidx)
                    gidx+=1

    # padding align
    for l in G:
        while len(l)<max(3,N-1):
            l.append(l[-1])
    return G

def generate_all_graph(MAXN=30):
    Gs=[]
    Gs.extend([[],[]])
    for n in range(2,MAXN+1):
        Gs.append(generate_graph(n))
    return Gs

# 上三角和下三角的坐标表示
# 获取z需要的坐标，
def generate_Tri(MAXN=30):
    uTri=[]
    lTri=[]
    for n in range(0,MAXN+1):
        if n==0:
            uTri.append([])
            lTri.append([])
            continue
        utmp=[]
        ltmp=[]
        for u in range(n):
            for v in range(u+1,n):
                utmp.append([u,v])
                ltmp.append([v,u])
        uTri.append(utmp)
        lTri.append(ltmp)
    return uTri,lTri

# h从y,z中获取feature 需要的坐标
def generate_h_cord(MAXN=30):
    h_cord=[]
    h_cord.extend([[],[]])
    for N in range(2,MAXN+1):
        tmp=[[*c,i+N]for i,c in enumerate(combinations(range(N),2))]
        h_cord.append(tmp)
    return h_cord

# g从z中获取feature 需要的坐标
def generate_g_cord(MAXN=30):
    g_cord=[]
    g_cord.extend([[],[],[]])
    for N in range(3,MAXN+1):
        h_id={c:i+N for i,c in enumerate(combinations(range(N),2))}  #id of z
        tmp=[]
        for i in range(N):
            for j in range(i+1,N):
                for k in range(j+1,N):
                    z1,z2,z3=h_id[(i,j)],h_id[(i,k)],h_id[(j,k)]
                    tmp.append([z1,z2,z3])
        g_cord.append(tmp)
    return g_cord

Gs=generate_all_graph()
uTri,lTri=generate_Tri()
h_cord=generate_h_cord()
g_cord=generate_g_cord()

# N*(N-1)->N*(N-1)/2,获取上三角特征，即获取z node feature
def get_z_nodefeature(mat,N):
    device=mat.device
    mat=unpackEdgeFeature(mat,N)
    uidx=torch.Tensor(uTri[N]).to(device).long()
    return mat[uidx[:,0],uidx[:,1],:]

def get_h_factorfeature(nodefeature,N):
    device=nodefeature.device
    h_cord_n=torch.Tensor(h_cord[N]).to(device).long()
    h_f=nodefeature[h_cord_n]
    return torch.mean(h_f,dim=1)

def get_g_factorfeature(nodefeature,N):
    device=nodefeature.device
    g_cord_n=torch.Tensor(g_cord[N]).to(device).long()
    g_f=nodefeature[g_cord_n]
    return torch.mean(g_f,dim=1)

# 计算图的边权重
def get_edgefeature(nodefeature,factorfeature,N):
    nff=torch.cat((nodefeature,factorfeature),dim=0)# node factor feature
    device=nodefeature.device
    graph=torch.Tensor(Gs[N]).to(device).long()
    ef=torch.cat((nff.unsqueeze(1).repeat((1,max(3,N-1),1)),nff[graph]),dim=-1)# edge feature
    return ef

class BasenetFgnnMeanfield(nn.Module):
    """
    main module of base model
    """
    def __init__(self, cfg):
        super(BasenetFgnnMeanfield, self).__init__()
        self.cfg=cfg

        # factor graph para
        NDIM=128# node feature dim
        FDIM=NDIM# factor feature dim
        
        self.action_ffn=nn.Sequential(nn.Linear(cfg.num_actions,NDIM//2),
                                      nn.ReLU(),
                                      nn.Linear(NDIM//2,NDIM),
                                      nn.LayerNorm([NDIM]),
                                      nn.ReLU())
        self.inter_ffn=nn.Sequential(nn.Linear(2,NDIM//2),
                                     nn.ReLU(),
                                     nn.Linear(NDIM//2,NDIM),
                                     nn.LayerNorm([NDIM]),
                                     nn.ReLU())
        
        self.action_fc=nn.Sequential(nn.Linear(NDIM,NDIM),
                                     nn.ReLU(),
                                     nn.Linear(NDIM,cfg.num_actions))
        self.inter_fc=nn.Linear(NDIM,2)

        self.fgnn=factor_mpnn(NDIM,[FDIM],
                              [64*2,64*2,128*2,128*2,256*2,256*2,128*2,128*2,64*2,64*2,NDIM],
                              [16])
        
        self.fc_edge=nn.Linear(2*NDIM,16)

        self.lambda_h=nn.Parameter(torch.Tensor([cfg.lambda_h]))
        self.lambda_g=nn.Parameter(torch.Tensor([cfg.lambda_g]))

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        
        # if self.cfg.pretrained!=None:
        #     for para in self.parameters():
        #         para.requires_grad=False
                
        # self.lambda_h.requires_grad=True
        # self.lambda_g.requires_grad=True
            

    def forward(self,batch_data):
        rank=dist.get_rank()
        act_score,inter_score,n_person=batch_data
        act_score=act_score.squeeze()
        inter_score=inter_score.squeeze()
        act_score=self.action_ffn(act_score)
        inter_score=self.inter_ffn(inter_score)
        N=n_person
        
        uidx=torch.Tensor(uTri[N]).long().to(rank)
        lidx=torch.Tensor(lTri[N]).long().to(rank)
        
        nodefeature=torch.cat((act_score,
                                get_z_nodefeature(inter_score,N)),dim=0)
        
        if N>=2:
            if N==2:factorfeature=get_h_factorfeature(nodefeature,N)
            else :
                factorfeature=torch.cat((get_h_factorfeature(nodefeature,N),
                                            get_g_factorfeature(nodefeature,N)),dim=0)
            weight=self.fc_edge(get_edgefeature(nodefeature,factorfeature,N)).unsqueeze(0)
            weight=F.relu(weight)
            graph=torch.Tensor(Gs[N]).unsqueeze(0).long().to(rank)
            nodefeature=nodefeature.transpose(0,1).unsqueeze(0).unsqueeze(-1)
            factorfeature=factorfeature.transpose(0,1).unsqueeze(0).unsqueeze(-1)

            nodefeature,factorfeature=self.fgnn(nodefeature,[factorfeature],[[graph,weight]])

            nodefeature=nodefeature.squeeze()
            nodefeature=nodefeature.transpose(0,1)
            actn_node_score=nodefeature[:N,:]
            E=actn_node_score.shape[-1]
            interaction_score=torch.zeros((N,N,E)).to(rank)
            interaction_score[uidx[:,0],uidx[:,1],:]=nodefeature[N:,:]
            interaction_score[lidx[:,0],lidx[:,1],:]=nodefeature[N:,:]
            interaction_score=packEdgeFeature(interaction_score,N)
            
            act_score=self.action_fc(act_score+actn_node_score)
            inter_score=self.inter_fc(interaction_score)
            
        else:
            act_score=self.action_fc(act_score)
            inter_score=self.inter_fc(inter_score)

        act_score,inter_score=meanfield(act_score, inter_score,
                                        self.cfg.knowledge,self.lambda_h,self.lambda_g)
        
        return act_score,inter_score
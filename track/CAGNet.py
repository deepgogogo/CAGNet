import torch 
import torch.nn as nn
import torch.nn.functional as F 


import numpy as np
from itertools import combinations

from backbone import get_backbone
from utils import *
from torchvision.ops import *

from mpnn import factor_mpnn
from utils import unpackEdgeFeature
import torch.distributed as dist
import functools
from meanfield import meanfield


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

def generate_all_graph(MAXN=15):
    Gs=[]
    Gs.extend([[],[]])
    for n in range(2,MAXN+1):
        Gs.append(generate_graph(n))
    return Gs

# 上三角和下三角的坐标表示
# 获取z需要的坐标，
def generate_Tri(MAXN=15):
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
def generate_h_cord(MAXN=15):
    h_cord=[]
    h_cord.extend([[],[]])
    for N in range(2,MAXN+1):
        tmp=[[*c,i+N]for i,c in enumerate(combinations(range(N),2))]
        h_cord.append(tmp)
    return h_cord

# g从z中获取feature 需要的坐标
def generate_g_cord(MAXN=15):
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

class MotionFgnn(nn.Module):
    
    def __init__(self,input_sz=1024,node_dim=128,factor_dim=128,edge_dim=16):
        super(MotionFgnn,self).__init__()
        
        self.fgnn=factor_mpnn(node_dim,[factor_dim],
                        [64*2,64*2,128*2,128*2,256*2,256*2,128*2,128*2,64*2,64*2,node_dim],
                        [edge_dim])
        
        self.edge_fc=nn.Sequential(nn.Linear(2*node_dim,edge_dim),
                                   nn.ReLU())
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                
    
    @functools.lru_cache(maxsize=100)
    def get_graph(self,n):
        assert n>1
        cn2=n*(n-1)//2
        g=[[] for _ in range(n+cn2)]
        # g[-1].extend([i for i in range(n)])
        for i,(u,v) in enumerate(combinations(range(n),2)):
            g[i+n].extend([u,v])
            g[u].append(i+n)
            g[v].append(i+n)
            
        for l in g:
            while len(l)<2:
                l.append(l[-1])

        return g
    
    def get_edge_feats(self, node_feats, factor_feats):
        rank=dist.get_rank()
        n=len(node_feats)
        nff=torch.cat((node_feats,factor_feats),dim=0)# node factor feature
        graph=torch.tensor(self.get_graph(n)).to(rank).long()
        tnff=nff.unsqueeze(1).repeat((1,2,1))
        gf=nff[graph]
        ef=torch.cat((nff.unsqueeze(1).repeat((1,2,1)),nff[graph]),dim=-1)# edge feature
        return ef
    
    def get_factor_feats(self, node_feats, n):
        idx=torch.tensor(list(combinations(range(n),2)),device=dist.get_rank())
        return (node_feats[idx[:,0]]+node_feats[idx[:,1]])/2
        
    
    def forward(self, node_feats):
        """ motion feats shape is [N,node_dim]
        """
        rank=dist.get_rank()
        n=len(node_feats)
        assert n>1
        graph=torch.tensor(self.get_graph(n),device=rank).unsqueeze(0).long()
        # factor_feats=torch.mean(node_feats,dim=0,keepdim=True)
        factor_feats=self.get_factor_feats(node_feats,n)
        edge_feats=self.edge_fc(self.get_edge_feats(node_feats,factor_feats)).unsqueeze(0)
        node_feats=node_feats.transpose(0,1).unsqueeze(0).unsqueeze(-1)
        factor_feats=factor_feats.transpose(0,1).unsqueeze(0).unsqueeze(-1)
        node_out,_=self.fgnn(node_feats,[factor_feats],[[graph,edge_feats]])
        node_out=node_out.squeeze().transpose(0,1)
        return node_out

class BasenetFgnnMeanfield(nn.Module):
    """
    main module of base model for CI dataset
    """
    def __init__(self, cfg):
        super(BasenetFgnnMeanfield, self).__init__()
        self.cfg=cfg

        D=self.cfg.emb_features #emb_features=1056   #output feature map channel of backbone
        K=self.cfg.crop_size[0] #crop_size = 5, 5  #crop size of roi align
        NFB=self.cfg.num_features_boxes #num_features_boxes = 1024
        # factor graph para
        NDIM=128# node feature dim
        NMID=64 # node feature mid layer dim
        FDIM=NDIM# factor feature dim

        self.backbone = get_backbone('inception')

        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.dropout_emb_1 = nn.Dropout(p=self.cfg.train_dropout_prob)
        
        self.fc_action_node=nn.Linear(NFB,NDIM)
        self.fc_action_mid=nn.Linear(NDIM,NMID)
        self.nl_action_mid=nn.LayerNorm([NMID])
        self.fc_action_final=nn.Linear(NMID,self.cfg.num_actions)

        self.fc_interactions_mid=nn.Linear(2*NFB,NDIM)#
        self.fc_interactions_final=nn.Linear(NDIM,2)

        self.fgnn=factor_mpnn(NDIM,[FDIM],
                              [64*2,64*2,128*2,128*2,256*2,256*2,128*2,128*2,64*2,64*2,NDIM],
                              [16])
        self.fc_edge=nn.Linear(2*NDIM,16)
        
        self.motion_fgnn=MotionFgnn()

            
        self.lambda_h=nn.Parameter(torch.Tensor([self.cfg.lambda_h]))
        self.lambda_g=nn.Parameter(torch.Tensor([self.cfg.lambda_g]))


        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                
        # for p in self.parameters():
        #     p.requires_grad=False
        
        for p in self.backbone.parameters():
            p.requires_grad=False
            
        # for p in self.motion_fgnn.parameters():
        #     p.requires_grad=True
            
        # self.lambda_h.requires_grad=True
        # self.lambda_g.requires_grad=True
            

    def forward(self,samples):
        frame_volumn=samples['frame_volumn']
        track_id_volumn=samples['track_id_volumn']
        box_num_volumn=samples['box_num_volumn']
        bbox_volumn=samples['bbox_volumn'].float()
        key_frame=samples['key_frame'].squeeze(-1)
        
        # read config parameters
        B,T=frame_volumn.shape[:2]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        MAX_N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes
        rank=dist.get_rank()
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        
        # Reshape the input data
        images_in_flat=torch.reshape(frame_volumn,(B*T,3,H,W))  #比如：torch.Size([16, 1, 3, 480, 720])->torch.Size([16, 3, 480, 720])
                
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat=prep_images(images_in_flat) #[归一化至[-1,1]]1.images = images.div(255.0) 2.images = torch.sub(images,0.5) 3.images = torch.mul(images,2.0)
        outputs=self.backbone(images_in_flat) 
            
    
        # Build multiscale features
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]): # outputs[0]等于的；outputs[1]不等于
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True) #双线性插值，将后面两维扩充成(57,87)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T（batchSize）, D, OH, OW，比如torch.Size([15, 1056, 57, 87])
        
        boxes_in_flat=torch.reshape(bbox_volumn,(B*T*MAX_N,4))  #B*T*MAX_N, 4，比如：(15*13,4)
        boxes_in_flat=boxes_in_flat.float() 
        boxes_in_flat=boxes_in_flat*torch.tensor([OW,OH,OW,OH],device=rank).float() # scale box
         
        #下面三行结束后，得到boxes_idx_flat，一个一维的tensor，[0,0,0,0,...,0,1,1,1,...,1,2,2,2...2.....14]，也就是0~batchsize-1，每个重复MAX_N遍
        boxes_idx=[i * torch.ones(MAX_N, dtype=torch.int)   for i in range(B*T) ] #共batchsize个tensor，每个tensor是一个MAX_N维的一维向量
        boxes_idx=torch.stack(boxes_idx).to(rank)  # B*T, MAX_N #从上一步而来，转变成batchsize*MAX_N的二维矩阵
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*MAX_N,))  #B*T*MAX_N,

        boxes_idx_flat=boxes_idx_flat.float()
        boxes_idx_flat=torch.reshape(boxes_idx_flat,(-1,1))

        # RoI Align       e.g. boxes_features_all：torch.Size([195, 1056, 5, 5])
        boxes_in_flat.requires_grad=False 
        boxes_idx_flat.requires_grad=False
        boxes_features_all=roi_align(features_multiscale,#比如：torch.Size([15, 1056, 57, 87])
                                            torch.cat((boxes_idx_flat,boxes_in_flat),1),#比如：torch.Size([195, 5])
                                            (5,5)) 
        
        boxes_features_all=boxes_features_all.reshape(B,T,MAX_N,-1)  #B,T,MAX_N, D*K*K 

        # Embedding 
        boxes_features_all=self.fc_emb_1(boxes_features_all)  # B,T,MAX_N, NFB
        boxes_features_all=F.relu(boxes_features_all)
        boxes_features_all=self.dropout_emb_1(boxes_features_all)# B,T,MAX_N,NFB
        actn_scores=self.fc_action_node(boxes_features_all)  
        actn_scores=F.relu(actn_scores) #B,T,MAX_N, NDIM
        
        motions=actn_scores.clone().detach()
        for b in range(B):
            track_id=track_id_volumn[b]
            for id in range(1,MAX_N+1):
                m=track_id==id
                if m.sum()<=1:continue
                motions[b,m]=self.motion_fgnn(motions[b,m])
                pass
        
    
        actions_scores=[]
        interaction_scores=[]
        
        for b,t in enumerate(key_frame):

            N=box_num_volumn[b,t]
            uidx=torch.Tensor(uTri[N]).long().to(rank)
            lidx=torch.Tensor(lTri[N]).long().to(rank)
            boxes_states_flat=boxes_features_all[b,t,:N,:].reshape(N,NFB)  #1,N,NFB
            # actn_score=self.fc_action_node(boxes_states_flat)  #N, NDIM
            # actn_score=F.relu(actn_score)
            actn_score=actn_scores[b,t,:N,:]
            
            interaction_flat=[]
            for i in range(N):
                for j in range(N):
                    if i!=j:
                        # concatenate features of two nodes
                        interaction_flat.append(torch.cat([boxes_states_flat[i],boxes_states_flat[j]],dim=0))
            interaction_flat=torch.stack(interaction_flat,dim=0) #N(N-1),NFB
            
            # ===== fgnn procedure
            interaction_flat=self.fc_interactions_mid(interaction_flat) #N*(N-1),NDIM
            interaction_flat=F.relu(interaction_flat)

            nodefeature=torch.cat((actn_score,
                                get_z_nodefeature(interaction_flat,N)),dim=0)
            # if N>=2:
            if N>2:
                if N==2:factorfeature=get_h_factorfeature(nodefeature,N)
                else :
                    factorfeature=torch.cat((get_h_factorfeature(nodefeature,N),
                                            get_g_factorfeature(nodefeature,N)),dim=0)
                weight=self.fc_edge(get_edgefeature(nodefeature,factorfeature,N)).unsqueeze(0)
                weight=F.relu(weight)
                graph=torch.Tensor(Gs[N]).unsqueeze(0).to(rank).long()
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
                
                actn_score = self.fc_action_mid(actn_score + actn_node_score+motions[b,t,:N,:])
                actn_score=self.nl_action_mid(actn_score)
                actn_score=F.relu(actn_score)
                actn_score=self.fc_action_final(actn_score)
                interaction_score=self.fc_interactions_final(interaction_score)
            else:
                actn_score = self.fc_action_mid(actn_score+motions[b,t,:N,:])
                actn_score=self.nl_action_mid(actn_score)
                actn_score=F.relu(actn_score)
                actn_score=self.fc_action_final(actn_score)
                interaction_score=self.fc_interactions_final(interaction_flat) #N(N-1), 2
            # =====
            
            Q_y,Q_z=meanfield(self.cfg, actn_score, interaction_score,
                                self.lambda_h, self.lambda_g)

            actions_scores.append(Q_y)
            interaction_scores.append(Q_z)
            
            pass
            
        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        interaction_scores=torch.cat(interaction_scores,dim=0)
        
        return actions_scores,interaction_scores
        
        
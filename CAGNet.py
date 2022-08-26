import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np
from itertools import combinations

from backbone import get_backbone
from utils import *
from torchvision.ops import *

from meanfield import meanfield
from mpnn import factor_mpnn
from utils import unpackEdgeFeature

def generate_graph(N):
    """
    n[0,n-1],z[n,n+n*(n-1)/2-1],vertex node including action label and interaction label
    h[n+n*(n-1)/2,n+n*(n-1)-1], factor node
    g[n+n*(n-1),n+n*(n-1)+n*(n-1)*(n-2)/6)-1], factor node
    build the factor graph
    :param N: number of node
    :return: factor graph
    """

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
    """
    generate all graphs
    :param MAXN:
    :return:
    """
    Gs=[]
    Gs.extend([[],[]])
    for n in range(2,MAXN+1):
        Gs.append(generate_graph(n))
    return Gs

def generate_Tri(MAXN=15):
    """
    the ordered triangle element cordinates
    eg. N=3, upper triangle [[0,1],[0,2],[1,2]]
    the uTri and lTri are used to comnpute the index of z node
    :param MAXN:
    :return:
    """
    uTri=[]
    lTri=[]
    for n in range(0,MAXN+1):
        if n==0:
            uTri.append([])
            lTri.append([])
            continue
        utmp=[]
        ltmp=[]
        # ordered cordinates
        for u in range(n):
            for v in range(u+1,n):
                utmp.append([u,v])
                ltmp.append([v,u])
        uTri.append(utmp)
        lTri.append(ltmp)
    return uTri,lTri

def generate_h_cord(MAXN=15):
    """
    h_cord is used to retrieve the features in y and z
    :param MAXN:
    :return:
    """
    h_cord=[]
    h_cord.extend([[],[]])
    for N in range(2,MAXN+1):
        tmp=[[*c,i+N]for i,c in enumerate(combinations(range(N),2))]
        h_cord.append(tmp)
    return h_cord

def generate_g_cord(MAXN=15):
    """
    g_cord is used to retrieve the features in z
    :param MAXN:
    :return:
    """
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

def get_z_nodefeature(mat,N):
    """
    etrieve the upper triangle features, that is z node features,
    transform feature shape from N*(N-1) to N*(N-1)/2
    :param mat:
    :param N:
    :return:
    """
    device=mat.device
    mat=unpackEdgeFeature(mat,N)
    uidx=torch.Tensor(uTri[N]).to(device).long()
    return mat[uidx[:,0],uidx[:,1],:]

def get_h_factorfeature(nodefeature,N):
    """
    node features consist of action feature and interaction feature
    according to the h_cord
    the final feature are average of the features of y1,y2,z
    :param nodefeature:
    :param N:
    :return:
    """
    device=nodefeature.device
    h_cord_n=torch.Tensor(h_cord[N]).to(device).long()
    h_f=nodefeature[h_cord_n]
    return torch.mean(h_f,dim=1)

def get_g_factorfeature(nodefeature,N):
    device=nodefeature.device
    g_cord_n=torch.Tensor(g_cord[N]).to(device).long()
    g_f=nodefeature[g_cord_n]
    return torch.mean(g_f,dim=1)

def get_edgefeature(nodefeature,factorfeature,N):
    """
    compute the edge weight of the factor graph,
    :param nodefeature: corresponding to the node in fgnn
    :param factorfeature:corresponding to the factor node in fgnn
    :param N:
    :return:
    """
    nff=torch.cat((nodefeature,factorfeature),dim=0)# node factor feature
    device=nodefeature.device
    graph=torch.Tensor(Gs[N]).to(device).long()
    ef=torch.cat((nff.unsqueeze(1).repeat((1,max(3,N-1),1)),nff[graph]),dim=-1)# edge feature
    return ef

class BasenetFgnnMeanfield(nn.Module):

    def __init__(self, cfg):
        super(BasenetFgnnMeanfield, self).__init__()
        self.cfg=cfg

        D=self.cfg.emb_features # #output feature map channel of backbone
        K=self.cfg.crop_size[0] #crop_size = 5, 5, crop size of roi align
        NFB=self.cfg.num_features_boxes #num_features_boxes = 1024
        # factor graph para
        NDIM=128# node feature dim
        NMID=64 # node feature mid layer dim
        FDIM=NDIM# factor feature dim

        self.backbone = get_backbone(cfg.backbone)

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
                              [16])# fgnn 10 layers


        self.fc_edge=nn.Linear(2*NDIM,16)

        self.lambda_h=nn.Parameter(torch.Tensor([self.cfg.lambda_h]))
        self.lambda_g=nn.Parameter(torch.Tensor([self.cfg.lambda_g]))



        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        for p in self.backbone.parameters():
            p.require_grad=False


    def forward(self,batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data
        # read config parameters
        B=images_in.shape[0]
        T=images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        MAX_N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes
        device=images_in.device
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))
                
        # Use backbone to extract features of images_in
        # Pre-precess first,normalize the image
        images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone(images_in_flat)
            
    
        # Build multiscale features
        # get the target features before roi align
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)
        
        boxes_in_flat=torch.reshape(boxes_in,(B*T*MAX_N,4))
            
        boxes_idx=[i * torch.ones(MAX_N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*MAX_N,))

        boxes_idx_flat=boxes_idx_flat.float()
        boxes_idx_flat=torch.reshape(boxes_idx_flat,(-1,1))

        # RoI Align     boxes_features_all：[num_of_person,1056,5,5]
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features_all=roi_align(features_multiscale,
                                            torch.cat((boxes_idx_flat,boxes_in_flat),1),
                                            (5,5)) 
        
        boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,-1)  #B*T,MAX_N, D*K*K #比如：torch.Size([15, 13, 26400])

        # Embedding 
        boxes_features_all=self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all=F.relu(boxes_features_all)
        # B*T,MAX_N,NFB
        boxes_features_all=self.dropout_emb_1(boxes_features_all)
        
    
        actions_scores=[]
        interaction_scores=[]

        bboxes_num_in=bboxes_num_in.reshape(B*T,)
        for bt in range(B*T):
            # process one frame
            N=bboxes_num_in[bt]
            uidx=torch.Tensor(uTri[N]).long().to(device)
            lidx=torch.Tensor(lTri[N]).long().to(device)
            boxes_features=boxes_features_all[bt,:N,:].reshape(1,N,NFB)  #1,N,NFB
    
            boxes_states=boxes_features  

            NFS=NFB

            # Predict actions
            boxes_states_flat=boxes_states.reshape(-1,NFS)  #1*N, NFS
            actn_score=self.fc_action_node(boxes_states_flat)  #1*N, actn_num
            actn_score=F.relu(actn_score)


            # Predict interactions
            interaction_flat=[]
            for i in range(N):
                for j in range(N):
                    if i!=j:
                        # concatenate features of two nodes
                        interaction_flat.append(torch.cat([boxes_states_flat[i],boxes_states_flat[j]],dim=0))
            interaction_flat=torch.stack(interaction_flat,dim=0) #N(N-1),2048


            # ===== fgnn procedure
            interaction_flat=self.fc_interactions_mid(interaction_flat) #N*(N-1),num_action
            interaction_flat=F.relu(interaction_flat)

            # compute the node feature
            nodefeature=torch.cat((actn_score,
                                   get_z_nodefeature(interaction_flat,N)),dim=0)

            # the fgnn are valid when N>2
            if N>2:
                # compute the factor feature
                # if N==2:factorfeature=get_h_factorfeature(nodefeature,N)
                # else :factorfeature=torch.cat((get_h_factorfeature(nodefeature,N),
                #                          get_g_factorfeature(nodefeature,N)),dim=0)
                factorfeature=torch.cat((get_h_factorfeature(nodefeature,N),
                                         get_g_factorfeature(nodefeature,N)),dim=0)
                weight=self.fc_edge(get_edgefeature(nodefeature,factorfeature,N)).unsqueeze(0)
                weight=F.relu(weight)
                graph=torch.Tensor(Gs[N]).unsqueeze(0).to(device).long()
                nodefeature=nodefeature.transpose(0,1).unsqueeze(0).unsqueeze(-1)
                factorfeature=factorfeature.transpose(0,1).unsqueeze(0).unsqueeze(-1)

                nodefeature,factorfeature=self.fgnn(nodefeature,[factorfeature],[[graph,weight]])

                nodefeature=nodefeature.squeeze()
                nodefeature=nodefeature.transpose(0,1)
                actn_node_score=nodefeature[:N,:]
                E=actn_node_score.shape[-1]
                interaction_score=torch.zeros((N,N,E)).to(device)
                interaction_score[uidx[:,0],uidx[:,1],:]=nodefeature[N:,:]
                interaction_score[lidx[:,0],lidx[:,1],:]=nodefeature[N:,:]
                interaction_score=packEdgeFeature(interaction_score,N)# N*N=>N*(N-1)

                actn_score = self.fc_action_mid(actn_score + actn_node_score)
                actn_score=self.nl_action_mid(actn_score)
                actn_score=F.relu(actn_score)
                actn_score=self.fc_action_final(actn_score)
                interaction_score=self.fc_interactions_final(interaction_score)
            else:
                actn_score = self.fc_action_mid(actn_score)
                actn_score=self.nl_action_mid(actn_score)
                actn_score=F.relu(actn_score)
                actn_score=self.fc_action_final(actn_score)
                interaction_score=self.fc_interactions_final(interaction_flat) #N(N-1), 2
            # =====

            Q_y,Q_z=meanfield(self.cfg, actn_score, interaction_score,
                              self.lambda_h,self.lambda_g)
            actions_scores.append(Q_y)
            interaction_scores.append(Q_z)

        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        interaction_scores=torch.cat(interaction_scores,dim=0)

        return actions_scores,interaction_scores
        
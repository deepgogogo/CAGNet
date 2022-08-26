import torch
from itertools import combinations,groupby,permutations
from  utils import unpackEdgeFeature,packEdgeFeature,setSeed
import time

"""
N is the number of nodes
|H|=M=NC2
|G|=MC3
H(y,y,z), G(z ,z,z)
actionScore is a matrix with shape (N,C)
interactionScore is a matrix with shape(N*(N-1),2)
"""

# The conflict matrix
conflict={
    'bit':[                 # Here 1 indicates conflicts
        [1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,0],
        [1,1,0,1,1,1,1,1,1],
        [1,1,1,0,1,1,1,1,1],
        [1,1,1,1,0,1,1,1,1],
        [1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,0],
        [0,0,1,1,1,0,0,0,1]
    ],
    'ut':[                 # Here 1 indicates conflicts
        [0,1,1,1,1,1,1,1,1],
        [1,0,1,1,1,1,1,1,1],
        [1,1,1,0,1,1,1,1,1],
        [1,1,0,1,1,1,1,1,1],
        [1,1,1,1,1,0,1,1,1],
        [1,1,1,1,0,1,1,1,1],
        [1,1,1,1,1,1,1,0,1],
        [1,1,1,1,1,1,0,1,1],
        [1,1,1,1,1,1,1,1,1]
    ],
    'tvhi':[                 # Here 1 indicates conflicts
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1]
    ]
}

MAXN=10 # At most 10 people

H=[] # list of lists
H_map=[] # list of map for the convenience of fetching H-clusters a node belongs to
def generate_HSet():
    def left(p):
        return p[0]
    def right(p):
        return p[1]
    global H_map,H
    H.append([])
    H.append([])
    H_map=[{}for i in range(MAXN+1)]
    for n in range(2,MAXN+1):
        tmp_comb = [[*c,i] for i,c in enumerate(combinations(range(n), 2))]
        H.append(tmp_comb)
        d={}
        for k,l in groupby(tmp_comb,key=left):
            d[k]=list(l)
        tmp_comb=sorted(tmp_comb,key=right)
        for k,l in groupby(tmp_comb,key=right):
            if k not in d.keys():
                d[k]=[]
            d[k]+=list(l)

        H_map[n]=d

G=[] # list of lists
G_map=[]# # list of maps for the convenience of fetching G-clusters a node belongs to
def generate_GSet():
    def left(tp):
        return tp[0]
    def middle(tp):
        return tp[1]
    def right(tp):
        return tp[2]
    global G,G_map
    G_map=[{} for i in range(MAXN+1)]
    for n in range(MAXN+1):
        m=n*(n-1)//2
        tmp_comb = [c for c in combinations(range(m), 3)]
        G.append(tmp_comb)
        d={}
        for k,l in groupby(tmp_comb,key=left):
            d[k]=list(l)
        tmp_comb=sorted(tmp_comb,key=middle)
        for k,l in groupby(tmp_comb,key=middle):
            if k not in d.keys():
                d[k]=[]
            d[k]+=list(l)
        tmp_comb=sorted(tmp_comb,key=right)
        for k,l in groupby(tmp_comb,key=right):
            if k not in d.keys():
                d[k]=[]
            d[k]+=list(l)
        G_map[n]=d

Z_map=[] # space [0,N*(N-1)) -> space [0,N*(N-1)/2)
Z_map_r=[] # space [0,N*(N-1)/2) -> space [0,N*(N-1))

###
# output a list Z_map of n*(n-1)/2 entries,
# suppose N = permuitation(range(n),2), and the i-th entry Z_map[i] = [a,b],
# then: N[a] and N[b] give the same ids of y nodes but with reversed order, and
# N[a][0] < N[a][1]
#
def generate_Z_map():
    global Z_map,Z_map_r
    for n in range(MAXN+1):
        d={}
        l=[]
        cmb={}
        for i,c in enumerate(combinations(range(n),2)):
            cmb[c]=i
        for i,a in enumerate(permutations(range(n),2)):
            d[a]=i
        r = [0 for i in range(n * (n - 1))]
        for k,v in d.items():
            if k[0]<k[1]:
                t=(k[1],k[0])
                l.append([v,d[t]])
                r[v]=cmb[k]
                r[d[t]]=cmb[k]
        Z_map.append(l)
        Z_map_r.append(r)

uTri=[]
lTri=[]
def generate_Tri():
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

def meanfield(cfg,actn_score,intrct_score,lambda_h,lamda_g,C=True,T=True):

    cur_actn_score = actn_score
    cur_intrct_score = intrct_score
    # initialization
    num_msg_time=cfg.num_msg_time
    for t in range(num_msg_time):
        Q_y = torch.softmax(cur_actn_score, dim=1)
        Q_z = torch.softmax(cur_intrct_score, dim=1)
        cur_actn_score, cur_intrct_score = one_step(cfg,Q_y,Q_z,lambda_h,lamda_g)
        cur_actn_score = actn_score - cur_actn_score
        cur_intrct_score = intrct_score - cur_intrct_score

    return cur_actn_score, cur_intrct_score

def one_step(cfg,Qy, Qz, lambda_h, lamda_g):
    # Update Q(y)
    global Z_map,Z_map_r
    device=Qy.device
    mask=torch.Tensor(conflict[cfg.dataset_name]).to(device).long()
    N,C=Qy.shape[0],Qy.shape[1]

    m=torch.matmul(Qy.reshape(N,1,C,1), Qy.reshape(1,N,1,C))#(N,N,C,C)
    m=m*mask
    H_map_N = H_map[N]
    Z_map_N = torch.Tensor(Z_map[N]).to(device).long()

    y_score_new=torch.zeros_like(Qy).to(Qy.device)
    tQy=Qy.unsqueeze(1).repeat((1,C,1))*mask # N,C,C

    for id in range(N):
        idm=torch.Tensor(H_map_N[id]).to(device).long()
        h1s=torch.sum(tQy[idm[:,0],:,:],dim=2)*Qz[Z_map_N[idm[:,2],0],1].unsqueeze(-1)*(idm[:,0]!=id).unsqueeze(-1)
        h2s=torch.sum(tQy[idm[:,1],:,:],dim=2)*Qz[Z_map_N[idm[:,2],0],1].unsqueeze(-1)*(idm[:,1]!=id).unsqueeze(-1)
        cls_score=torch.sum(h1s,dim=0)+torch.sum(h2s,dim=0)
        y_score_new[id]=cls_score*lambda_h

    # Update Q(z)
    poly_h=torch.zeros_like(Qz).to(Qz.device)
    h=torch.Tensor(H[N]).long()
    cc=m[h[:,0],h[:,1],:,:]
    poly_h[Z_map_N[h[:,2]],1]=torch.sum(cc,dim=[1,2]).unsqueeze(-1)*lambda_h
    z_score_new = poly_h

    mQz=unpackEdgeFeature(Qz,N)
    m_poly_g=torch.zeros_like(mQz).to(device)
    mesh0=torch.Tensor([[0,0],[0,1]]).to(device)
    mesh1=torch.Tensor([[0,1],[1,0]]).to(device)

    if N > 2:
        uidx=torch.Tensor(uTri[N]).to(device).long()
        lidx=torch.Tensor(lTri[N]).to(device).long()
        row=mQz.unsqueeze(-1).unsqueeze(1)
        col=mQz.transpose(0,1).unsqueeze(2).unsqueeze(0)
        t=torch.matmul(row,col)
        m_poly_g[:,:,0]=torch.sum(t*mesh0,dim=[2,3,4])
        m_poly_g[:,:,1]=torch.sum(t*mesh1,dim=[2,3,4])
        m_poly_g[lidx[:,0],lidx[:,1],:]=m_poly_g[uidx[:,0],uidx[:,1],:]

        poly_g=packEdgeFeature(m_poly_g,N)*lamda_g
        z_score_new += poly_g

    return y_score_new, z_score_new

generate_HSet()
generate_GSet()
generate_Z_map()
generate_Tri()

if __name__=='__main__':
    N=10
    setSeed(20)
    score_y=torch.randn((N,9))# assume this is energy
    score_z=torch.randn((N*(N-1),2))
    #
    tstart=time.time()
    y_score,z_score = meanfield(score_y,score_z,0.5,0.1)
    tend = time.time()
    print('y prob:')
    print(torch.softmax(y_score,dim=1))
    print('z prob:')
    print(torch.softmax(z_score,dim=1))
    print(tend-tstart)


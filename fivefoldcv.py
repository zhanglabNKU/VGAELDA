import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from models import GraphConv, AE, LP
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Dimension of representations')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight between lncRNA space and disease space')
parser.add_argument('--data', type=int, default=1, choices=[1,2],
                    help='Dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

set_seed(args.seed,args.cuda)
gdi, ldi, rnafeat, gl, gd = load_data(args.data,args.cuda)

class GNNq(nn.Module):
    def __init__(self):
        super(GNNq,self).__init__()
        self.gnnql = AE(rnafeat.shape[1],256,args.hidden)
        self.gnnqd = AE(gdi.shape[0],256,args.hidden)
    
    def forward(self,xl0,xd0):
        hl,stdl,xl = self.gnnql(gl,xl0)
        hd,stdd,xd = self.gnnqd(gd,xd0)
        return hl,stdl,xl,hd,stdd,xd

class GNNp(nn.Module):
    def __init__(self):
        super(GNNp,self).__init__()
        self.gnnpl = LP(args.hidden,ldi.shape[1])
        self.gnnpd = LP(args.hidden,ldi.shape[0])

    def forward(self,y0):
        yl,zl = self.gnnpl(gl,y0)
        yd,zd = self.gnnpd(gd,y0.t())
        return yl,zl,yd,zd

print("Dataset {}, 5-fold CV".format(args.data))

def criterion(output,target,msg,n_nodes,mu,logvar):
    if msg == 'disease':
        cost = F.binary_cross_entropy(output,target)
    else:
        cost = F.mse_loss(output,target)
    
    KL = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KL

def train(gnnq,gnnp,xl0,xd0,y0,epoch,alpha):
    beta0 = 1.0
    gamma0 = 1.0
    optp = torch.optim.Adam(gnnp.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    optq = torch.optim.Adam(gnnq.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    for e in range(epoch):
        gnnq.train()
        hl,stdl,xl,hd,stdd,xd = gnnq(xl0,xd0)
        lossql = criterion(xl,xl0,
            "lncrna",gl.shape[0],hl,stdl)
        lossqd = criterion(xd,xd0,
            "disease",gd.shape[0],hd,stdd)
        lossq = alpha*lossql + (1-alpha)*lossqd + beta0*e*F.mse_loss(
            torch.mm(hl,hd.t()),y0)/epoch
        optq.zero_grad()
        lossq.backward()
        optq.step()
        gnnq.eval()
        with torch.no_grad():
            hl,_,_,hd,_,_ = gnnq(xl0,xd0)
        
        gnnp.train()
        yl,zl,yd,zd = gnnp(y0)
        losspl = F.binary_cross_entropy(yl,y0) + gamma0*e*F.mse_loss(zl,hl)/epoch
        losspd = F.binary_cross_entropy(yd,y0.t()) + gamma0*e*F.mse_loss(zd,hd)/epoch
        lossp = alpha*losspl + (1-alpha)*losspd
        optp.zero_grad()
        lossp.backward()
        optp.step()

        gnnp.eval()
        with torch.no_grad():
            yl,_,yd,_ = gnnp(y0)
        
        if e%20 == 0:
            print('Epoch %d | Lossp: %.4f | Lossq: %.4f' % (e, lossp.item(),lossq.item()))
        
    return alpha*yl+(1-alpha)*yd.t()

def fivefoldcv(A,alpha=0.5):
    N = A.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    res = torch.zeros(5,A.shape[0],A.shape[1])
    aurocl = np.zeros(5)
    auprl = np.zeros(5)
    for i in range(5):
        print("Fold {}".format(i+1))
        A0 = A.clone()
        for j in range(i*N//5,(i+1)*N//5):
            A0[idx[j],:] = torch.zeros(A.shape[1])
        
        gnnq = GNNq()
        gnnp = GNNp()
        if args.cuda:
            gnnq = gnnq.cuda()
            gnnp = gnnp.cuda()

        train(gnnq,gnnp,rnafeat,gdi.t(),A0,args.epochs,args.alpha)
        gnnq.eval()
        gnnp.eval()
        yli,_,ydi,_ = gnnp(A0)
        resi = alpha*yli + (1-alpha)*ydi.t()
        #resi = scaley(resi)
        res[i] = resi
        
        if args.cuda:
            resi = resi.cpu().detach().numpy()
        else:
            resi = resi.detach().numpy()
        
        auroc,aupr = show_auc(resi,args.data)
        aurocl[i] = auroc
        auprl[i] = aupr
        
    ymat = res[auprl.argmax()]
    if args.cuda:
        return ymat.cpu().detach().numpy()
    else:
        return ymat.detach().numpy()

title = 'result--dataset'+str(args.data)
ymat = fivefoldcv(ldi,alpha=args.alpha)
title += '--fivefoldcv'
ymat = scaley(ymat)
np.savetxt(title+'.txt',ymat,fmt='%10.5f',delimiter=',')
print("===Final result===")
show_auc(ymat,args.data)
import numpy as np
import torch
import argparse
from sklearn.preprocessing import minmax_scale,scale
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc

def scaley(ymat):
    return (ymat-ymat.min())/ymat.max()

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def load_data(data,cuda):
    path = 'Dataset'+str(data)
    gdi = np.loadtxt(path + '/known_gene_disease_interaction.txt')
    ldi = np.loadtxt(path + '/known_lncRNA_disease_interaction.txt')
    rnafeat = np.loadtxt(path + '/rnafeat.txt', delimiter=',')
    rnafeat = minmax_scale(rnafeat,axis=0)
    gdit = torch.from_numpy(gdi).float()
    ldit = torch.from_numpy(ldi).float()
    rnafeatorch = torch.from_numpy(rnafeat).float()
    gl = norm_adj(rnafeat)
    gd = norm_adj(gdi.T)
    if cuda:
        gdit = gdit.cuda()
        ldit = ldit.cuda()
        rnafeatorch = rnafeatorch.cuda()
        gl = gl.cuda()
        gd = gd.cuda()
    
    return gdit, ldit, rnafeatorch, gl, gd

def neighborhood(feat,k):
    # compute C
    featprod = np.dot(feat.T,feat)
    smat = np.tile(np.diag(featprod),(feat.shape[1],1))
    dmat = smat + smat.T - 2*featprod
    dsort = np.argsort(dmat)[:,1:k+1]
    C = np.zeros((feat.shape[1],feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i,j] = 1.0
    
    return C

def normalized(wmat):
    deg = np.diag(np.sum(wmat,axis=0))
    degpow = np.power(deg,-0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow,wmat),degpow)
    return W

def norm_adj(feat):
    C = neighborhood(feat.T,k=10)
    norm_adj = normalized(C.T*C+np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g

def show_auc(ymat,data):
    path = 'Dataset'+str(data)
    ldi = np.loadtxt(path + '/known_lncRNA_disease_interaction.txt')
    y_true = ldi.flatten()
    ymat = ymat.flatten()
    fpr,tpr,rocth = roc_curve(y_true,ymat)
    auroc = auc(fpr,tpr)
    #np.savetxt('roc.txt',np.vstack((fpr,tpr)),fmt='%10.5f',delimiter=',')
    precision,recall,prth = precision_recall_curve(y_true,ymat)
    aupr = auc(recall,precision)
    #np.savetxt('pr.txt',np.vstack((recall,precision)),fmt='%10.5f',delimiter=',')
    print('AUROC= %.4f | AUPR= %.4f' % (auroc,aupr))
    # rocdata = np.loadtxt('roc.txt',delimiter=',')
    # prdata = np.loadtxt('pr.txt',delimiter=',')
    # plt.figure()
    # plt.plot(rocdata[0],rocdata[1])
    # plt.plot(prdata[0],prdata[1])
    # plt.show()
    return auroc,aupr
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self,in_dim,out_dim,drop=0.5,bias=False,activation=None):
        super(GraphConv,self).__init__()
        self.dropout = nn.Dropout(drop)
        self.activation = activation
        self.w = nn.Linear(in_dim,out_dim,bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        self.bias = bias
        if self.bias:
            nn.init.zeros_(self.w.bias)
    
    def forward(self,adj,x):
        x = self.dropout(x)
        x = adj.mm(x)
        x = self.w(x)
        if self.activation:
            return self.activation(x)
        else:
            return x

class AE(nn.Module):
    def __init__(self,feat_dim,hid_dim,out_dim,bias=False):
        super(AE,self).__init__()
        self.conv1 = GraphConv(feat_dim,hid_dim,bias=bias,activation=F.relu)
        self.mu = GraphConv(hid_dim,out_dim,bias=bias,activation=torch.sigmoid)
        self.conv3 = GraphConv(out_dim,hid_dim,bias=bias,activation=F.relu)
        self.conv4 = GraphConv(hid_dim,feat_dim,bias=bias,activation=torch.sigmoid)
        self.logvar = GraphConv(hid_dim,out_dim,bias=bias,activation=torch.sigmoid)

    def encoder(self,g,x):
        x = self.conv1(g,x)
        h = self.mu(g,x)
        std = self.logvar(g,x)
        return h,std
    
    def decoder(self,g,x):
        x = self.conv3(g,x)
        x = self.conv4(g,x)
        return x
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self,g,x):
        mu,logvar = self.encoder(g,x)
        z = self.reparameterize(mu, logvar)
        return mu,logvar,self.decoder(g,z)

class LP(nn.Module):
    def __init__(self,hid_dim,out_dim,bias=False):
        super(LP,self).__init__()
        self.res1 = GraphConv(out_dim,hid_dim,bias=bias,activation=F.relu)
        self.res2 = GraphConv(hid_dim,out_dim,bias=bias,activation=torch.sigmoid)
    
    def forward(self,g,z):
        z = self.res1(g,z)
        res = self.res2(g,z)
        return res,z
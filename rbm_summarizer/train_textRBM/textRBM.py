import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable




class TextRBM(nn.Module):
    def __init__(self,
                n_vis=15000,
                n_hin=200,
                k=1,
                device='cpu'):
        super(TextRBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
        self.device=device

    def sample_bernoulli(self,p):
        # p can be a tensor of any size
        return F.relu(torch.sign(p - Variable(torch.rand(p.size())).to(self.device)))   # draw binary samples with p being the probablity of being 1, we can also use bernoulli, this implementation is just more bottom.
        # return torch.distributions.Bernoulli(p).sample()
    
    def sample_multinomial(self, p):
        # p should be a tensor of dimension (K,) with sum = 1
        # check my notes for detailed algorithm
        p = torch.cat((torch.Tensor([0]).to(self.device), p), 0)
        c = torch.cumsum(p, dim=0)
        g = F.relu(torch.sign(c - Variable(torch.rand(1)).to(self.device)))

        return (1 - g[:-1]) * g[1:]
        # return torch.distributions.Multinomial(1, p).sample()

    def v_to_h(self,v, d):
        p_h = torch.sigmoid(F.linear(v, self.W) + \
            torch.mm(d.float().unsqueeze(-1), self.h_bias.unsqueeze(0)))
        sample_h = self.sample_bernoulli(p_h)
        return p_h,sample_h

    def h_to_v(self,h, d):
        p_v = torch.nn.Softmax(dim=-1)(F.linear(h,self.W.t(),self.v_bias))
        sample_v = torch.stack([torch.stack([self.sample_multinomial(p_v_i) for _ in range(d_i)]).sum(0) for (p_v_i, d_i) in zip(p_v, d)])
        return p_v,sample_v
    
    def forward(self,v):
        with torch.no_grad():
            v = v.float()
            d = v.sum(-1).int()
            p_h1,h1 = self.v_to_h(v, d)
            
            h_ = h1
            for _ in range(self.k):
                p_v_,v_ = self.h_to_v(h_, d)
                p_h_,h_ = self.v_to_h(v_, d)
            
            return v,v_

    def free_energy(self,v):
        v = v.float()
        d = v.sum(-1).int()
        vbias_term = v.mv(self.v_bias) # torch.mv() performs a matrix-vector product
        wx_b = F.linear(v, self.W) + \
            torch.mm(d.float().unsqueeze(-1), self.h_bias.unsqueeze(0))
        # hidden_term = wx_b.exp().add(1).log().sum(1)
        hidden_term = F.softplus(wx_b).sum(1)  # for numerical stability

        return (-hidden_term - vbias_term).mean()

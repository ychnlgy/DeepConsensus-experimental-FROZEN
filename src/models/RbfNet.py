import torch

from .NormalInit import NormalInit

class RbfNet(torch.nn.Module, NormalInit):
    def __init__(self, headsize, bodysize, tailsize, eps=1e-8, **kwargs):
        super(RbfNet, self).__init__()
        
        self.eps = eps
        self.miu = torch.nn.Parameter(torch.Tensor(bodysize).normal_(mean=0, std=0.02))
        self.rad = torch.nn.Parameter(torch.Tensor(bodysize).normal_(mean=0, std=0.02))
        
        self.W1 = torch.nn.Linear(headsize, bodysize, bias=False)
        self.W2 = torch.nn.Linear(bodysize, tailsize, bias=False)
        
        self.init_weights(self.W1)
        self.init_weights(self.W2)
    
    def get_init_targets(self):
        return [torch.nn.Linear]
    
    def forward(self, X):
    
        n = len(X.size())
        c = [1]*(n-1) + [-1]
        miu = self.miu.view(c)
        rad = self.rad.view(c)
        
        X = self.W1(X)
        d = (X - miu)/(self.eps + rad)
        e = torch.exp(-d**2)
        X = self.W2(e)
        
        return X

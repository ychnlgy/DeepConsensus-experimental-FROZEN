import torch

from .NormalInit import NormalInit

class RbfNet(NormalInit):
    def __init__(self, headsize, bodysize, tailsize, **kwargs):
        super(RbfNet, self).__init__()
        print("Using rbf")
        
        self.miu = torch.nn.Parameter(torch.Tensor(1, bodysize).normal_(mean=0, std=0.02))
        self.rad = torch.nn.Parameter(torch.Tensor(1, bodysize).normal_(mean=0, std=0.02))
        
        self.W1 = torch.nn.Linear(headsize, bodysize, bias=False)
        self.W2 = torch.nn.Linear(bodysize, tailsize, bias=False)
        
        self.init_weights(self.W1)
        self.init_weights(self.W2)
    
    def get_init_targets(self):
        return [torch.nn.Linear]
    
    def forward(self, X):
        X = self.W1(X)
        d = (X - self.miu)/(self.eps + self.rad)
        e = torch.exp(-d**2)
        return self.W2(e)

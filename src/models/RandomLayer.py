import torch

class RandomLayer(torch.nn.Module):

    def __init__(self, *size):
        super(RandomLayer, self).__init__()
        self.size = size
        
        # dummy parameter so it can pretend 
        # to be a network and be optimized.
        self.p = torch.nn.Parameter(torch.ones(1))
    
    def forward(self, X):
        out = torch.rand(len(X), *self.size, requires_grad=True)
        return out.to(X.device)

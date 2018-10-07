import torch

class Mean(torch.nn.Module):

    def __init__(self, dim):
        super(Mean, self).__init__()
        self.dim = dim
    
    def forward(self, X):
        return X.mean(dim=self.dim)

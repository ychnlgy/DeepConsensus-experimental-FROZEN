import torch

class Permute(torch.nn.Module):
    
    def __init__(self, *axis):
        super(Permute, self).__init__()
        self.axis = axis
    
    def forward(self, X):
        return X.permute(self.axis)

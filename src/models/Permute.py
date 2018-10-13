import torch

class Permute(torch.nn.Module):
    
    def __init__(self, *pos):
        super(Permute, self).__init__()
        self.pos = pos
    
    def forward(self, X):
        return X.permute(self.pos)

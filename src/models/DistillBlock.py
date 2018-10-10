import torch

class DistillBlock(torch.nn.Module):

    def __init__(self, conv, pool):
        super(DistillBlock, self).__init__()
        self.conv = conv
        self.pool = pool
        
    def forward(self, X):
        out = self.conv(X)
        vec = self.pool(out)
        return out, vec

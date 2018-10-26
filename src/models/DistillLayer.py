import torch

class DistillLayer(torch.nn.Module):

    def __init__(self, conv, pool):
        super(DistillLayer, self).__init__()
        self.conv = conv
        self.pool = pool
    
    def forward(self, X):
        conv = self.conv(X)
        conv, pool = self.pool(conv)
        return conv, pool

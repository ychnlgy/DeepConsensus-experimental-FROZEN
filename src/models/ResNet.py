import torch

class ResNet(torch.nn.Module):

    def __init__(self, *blocks):
        super(ResNet, self).__init__()
        self.blocks = torch.nn.ModuleList(blocks)
    
    def forward(self, X):
        for X in self.iter_forward(X):
            continue
        return X
    
    def iter_forward(self, X):
        for block in self.blocks:
            X, output = block(X)
            if output:
                yield X

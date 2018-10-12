import torch

class DistillNet(torch.nn.Module):
    def __init__(self, *blocks):
        super(DistillNet, self).__init__()
        self.blocks = torch.nn.ModuleList(blocks)
    
    def forward(self, X, labels=None):
        vecs = []
        for block in self.blocks:
            X, vec = block(X, labels)
            vecs.append(vec)
        return torch.cat(vecs, dim=1) # N, C

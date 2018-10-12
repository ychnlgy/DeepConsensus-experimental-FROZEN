import torch

class DistillBlock(torch.nn.Module):

    def __init__(self, conv, pruner):
        super(DistillBlock, self).__init__()
        self.conv = conv
        self.pruner = pruner
        
    def forward(self, X, labels):
        out = self.conv(X)
        return self.pruner(out, labels)

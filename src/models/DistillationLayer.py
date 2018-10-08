import torch

PERMUTATION = (0, 2, 3, 1) # (N, W, H, C)

class DistillationLayer(torch.nn.Module):

    def __init__(self, cnn, lin):
        super(DistillationLayer, self).__init__()
        self.cnn = cnn
        self.lin = lin
        
    def forward(self, X):
        X = self.cnn(X).permute(PERMUTATION)
        X = self.lin(X).permute(PERMUTATION)
        return X

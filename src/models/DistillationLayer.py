import torch

class DistillationLayer(torch.nn.Module):

    def __init__(self, cnn, lin):
        super(DistillationLayer, self).__init__()
        self.cnn = cnn
        self.lin = lin
        
    def forward(self, X):
        X = self.cnn(X).permute(0, 2, 3, 1) # (N, W, H, C)
        X = self.lin(X).permute(0, 3, 1, 2) # (N, C, W, H)
        return X

import torch

class DistillNet(torch.nn.Module):

    def __init__(self, *layers):
        super(DistillNet, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, X):
        return sum(self.iter_forward(X))
        #return torch.cat(vecs, dim=1) # N, C1 + C2...
    
    def iter_forward(self, X):
        for layer in self.layers:
            X, vec = layer(X)
            yield vec

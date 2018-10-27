import torch

class DistillNet(torch.nn.Module):

    def __init__(self, *layers):
        super(DistillNet, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, X):
        vecs = sum(self.iter_forward(X))
        if not self.training:
            print(vecs.max(dim=1)[0])
        return vecs
        #return torch.cat(vecs, dim=1) # N, C1 + C2...
    
    def iter_forward(self, X):
        for layer in self.layers:
            X, vec = layer(X)
            yield vec

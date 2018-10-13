import torch

class DistillNet(torch.nn.Module):

    def __init__(self, *layers):
        super(DistillNet, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, X):
        vecs = []
        for layer in self.layers:
            X, vec = layer(X)
            vecs.append(vec)
        summary = torch.cat(vecs, dim=1) # N, C1 + C2 +...
        return summary

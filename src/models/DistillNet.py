import torch

class DistillNet(torch.nn.Module):

    def __init__(self, *layers, iternet=None):
        super(DistillNet, self).__init__()
        
        if iternet is not None:
            self.forward = self.iter_forward
            self.iternet = iternet
        
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, X):
        vecs = []
        for layer in self.layers:
            X, vec = layer(X)
            vecs.append(vec)
        return torch.cat(vecs, dim=1) # N, C1 + C2 + ...
    
    def iter_forward(self, X):
        vecs = []
        for i, (output, layer) in enumerate(zip(self.iternet, self.layers)):
            vecs.append(layer(output)[1])
        assert i + 1 == len(self.layers)
        return torch.cat(vecs, dim=1)

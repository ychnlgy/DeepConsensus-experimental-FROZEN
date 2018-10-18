import torch

import misc

class DistillNet(torch.nn.Module):

    def __init__(self, iternet, layers,encoder):
        super(DistillNet, self).__init__()
        self.iternet = iternet
        self.encoder = encoder
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, X):
        vecs = misc.util.reverse_iterator(self.generate_vecs(X))
        vecs = torch.stack(vecs) # layers, N, C
        encoded, state = self.encoder(vecs)
        return state[0] # N, C'
    
    def generate_vecs(self, X):
        Xs = self.iternet.iter_forward(X)
        for i, (output, layer) in enumerate(zip(Xs, self.layers)):
            yield layer(output)
        assert i + 1 == len(self.layers)

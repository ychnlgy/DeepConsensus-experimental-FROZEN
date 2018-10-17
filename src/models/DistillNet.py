import torch

import misc

class DistillNet(torch.nn.Module):

    def __init__(self, layers):
        super(DistillNet, self).__init__()
        
        #self.encoder = encoder
        
#        if iternet is not None:
#            self.generate_vecs = self.generate_vecs_iter
#            self.iternet = iternet
        
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, X):
        vecs = misc.util.reverse_iterator(self.generate_vecs(X))
        vecs = torch.stack(vecs) # layers, N, C
        return state[0] # N, C'
    
    def generate_vecs(self, X):
        for layer in self.layers:
            X, vec = layer(X)
            yield vec
    
#    def generate_vecs_iter(self, X):
#        Xs = self.iternet.iter_forward(X)
#        for i, (output, layer) in enumerate(zip(Xs, self.layers)):
#            yield layer(output)[1]
#        assert i + 1 == len(self.layers)

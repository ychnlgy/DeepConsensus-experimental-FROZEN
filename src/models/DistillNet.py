import torch

import misc

class DistillNet(torch.nn.Module):

    def __init__(self, *layers):
        super(DistillNet, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, X):
        vecs = sum(self.combine(X))
        misc.debug.println(vecs[0])
        return vecs
        #return torch.cat(vecs, dim=1) # N, C1 + C2...
    
    def iter_forward(self, X):
        for layer in self.layers:
            X, vec = layer(X)
            misc.debug.println(vec[0])
            yield vec
            
    def combine(self, X):
        it = self.iter_forward(X)
        return it
#        a = next(it)
#        b = next(it)
#        m = self.max(a)
#        n = self.max(b)
#        p = None
#        yield a * n
#        for c in it:
#            p = self.max(c)
#            yield (m + p)/2.0 * b
#            a, b = b, c
#            m, n = n, p
#        yield m * b

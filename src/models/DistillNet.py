import torch

class DistillNet(torch.nn.Module):

    def __init__(self, *layers):
        super(DistillNet, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.max = torch.nn.Softmax(dim=1)
    
    def forward(self, X):
        vecs = self.combine(X)
        return vecs
        #return torch.cat(vecs, dim=1) # N, C1 + C2...
    
    def iter_forward(self, X):
        for layer in self.layers:
            X, vec = layer(X)
            print(vec[0])
            yield vec
        print("")
    
    def combine(self, X):
        out = 0
        for vec in self.iter_forward(X):
            out = out + vec
        return out
            
#    def combine(self, X):
#        it = self.iter_forward(X)
#        a = next(it)
#        b = next(it)
#        m = self.max(a)
#        n = self.max(b)
#        p = None
#        yield a * n
#        for c in it:
#            p = self.max(c)
#            yield m * b * p
#            a, b = b, c
#            m, n = n, p
#        yield m * b

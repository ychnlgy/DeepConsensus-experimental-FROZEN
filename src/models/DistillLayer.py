import torch

import misc

PERMUTATION = (2, 3, 0, 1)

class DistillLayer(torch.nn.Module):

    def __init__(self, interpreter, pooler, summarizer):
        super(DistillLayer, self).__init__()
        self.intp = interpreter
        self.pool = pooler
        self.sumz = summarizer
    
    def forward(self, X):
        X = misc.matrix.apply_permutation(self.intp, X, PERMUTATION)
        X = self.pool(X)
        X = misc.matrix.apply_permutation(self.sumz, X, PERMUTATION)
        return X.contiguous()

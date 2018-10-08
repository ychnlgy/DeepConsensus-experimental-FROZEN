import torch

import misc

PERMUTATION = (2, 3, 0, 1) # (W, H, N, C)

class DistillationLayer(torch.nn.Module):

    def __init__(self, interpreter, pool, summarizer):
        super(DistillationLayer, self).__init__()
        self.interpreter = interpreter
        self.pool = pool
        self.summarizer = summarizer
        
    def forward(self, X):
        X = misc.matrix.apply_permutation(self.interpreter, X, PERMUTATION)
        X = self.pool(X)
        X = misc.matrix.apply_permutation(self.summarizer, X, PERMUTATION)
        return X.contiguous()

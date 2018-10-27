import torch

from .Norm import Norm

class SoftminNorm(Norm):

    def __init__(self):
        super(SoftminNorm, self).__init__()
        self.max = torch.nn.Softmax(dim=1)

    def reduce(self, vectors, targets):
        out = super(SoftminNorm, self).reduce(vectors, targets)
        return self.max(-out)

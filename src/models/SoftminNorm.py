import torch

from .Norm import Norm

class SoftminNorm(Norm):

    def __init__(self, *args, **kwargs):
        super(SoftminNorm, self).__init__(*args, **kwargs)
        self.min = torch.nn.Softmin(dim=1)

    def reduce(self, vectors, targets):
        out = super(SoftminNorm, self).reduce(vectors, targets)
        return self.min(out)
